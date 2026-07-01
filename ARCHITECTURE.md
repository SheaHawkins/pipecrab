We differ from pipecat in a few ways:

1. Pipecat Processor = Pipecrab Stage
1. The pipeline is split into two lanes: Data and Sys (high-priority). Messages in the sys queue are drained first and can interrupt work in progress, as well as travel to stage upstream (such as for configuration requests). The data lane is one-directional and flows downstream only.
1. A Stage can manage internal state via the uninterruptable synchronous `decide` function. By contrast, the async `perform` function can be interrupted but cannot modify state. This prevents broken state 
1. SystemFrames are distinct from DataFrames. In pipecrab, they don't share an inheritance tree so you can't accidentally push SystemFrames into the downstream-only data lane. 
1. `decide` returns a `Decision` (disposition + effects) instead of a bare effect list. `Decision::forward()` is the default, so an un-overridden stage is a transparent pass-through and you never have to re-push frames you don't touch. `Decision::drop().emit(x)` transforms a frame; `Decision::forward().emit(x)` taps it without consuming it. See the `Decision` rustdoc for all four forms.

## Audio I/O: the real-time bridge

Audio enters and leaves the pipeline through the `AudioSource` / `AudioSink` traits (`pipecrab-audio`). A concrete backend such as `pipecrab-audio-cpal` faces a constraint the rest of the runtime doesn't: the OS delivers and demands audio on a **real-time callback thread that must never block, allocate, or lock** — a missed deadline there is an audible glitch. But our pipeline is `async` and single-threaded. Reconciling those two worlds is the job of the `bridge` module.

**Primitives.** One [`rtrb`](https://docs.rs/rtrb) lock-free single-producer/single-consumer ring of `f32` per direction, plus a `Signal` — a `futures::task::AtomicWaker` and an `AtomicBool` "stream died" flag. The ring moves samples one way; the `Signal` moves wakeups the other way. Neither side ever locks or allocates.

```
capture:  mic ─▶ [RT input cb = producer] ─push f32─▶ (rtrb) ─pop chunk─▶ [CaptureRing = async consumer] ─▶ next_chunk()
                        │ wake                                                   ▲ register waker
                        └──────────────────────────── Signal ───────────────────┘

playback: play() ─▶ [PlaybackRing = async producer] ─push f32─▶ (rtrb) ─pop─▶ [RT output cb = consumer] ─▶ speaker
                        │ register waker                                         │ wake (freed room)
                        └──────────────────────────── Signal ───────────────────┘
```

**Capture.** The RT input callback is the ring's *producer*: it downmixes each frame to mono `f32`, pushes it, and calls `Signal::wake`. On the async side `CaptureRing::next_chunk` is the *consumer*: if a whole (~20 ms) chunk is buffered it pops it; otherwise it registers its task waker on the `Signal` and returns `Pending`, to be woken by the next callback. If the ring is full when the callback fires (the async side fell behind), the surplus samples are dropped and counted as an *overrun* (`CpalSource::overruns()`) rather than silently lost.

**Playback.** The roles flip. `PlaybackRing::play` is the *producer*: it pushes a chunk's samples and, if the ring fills, registers its waker and awaits — this backpressure paces the caller to real time. The RT output callback is the *consumer*: it pops one mono sample per frame (duplicating it across the device's channels), outputs silence on underrun, and calls `Signal::wake` after freeing room so a blocked `play` resumes.

**Failure & shutdown.** cpal's error callback calls `Signal::fail`, which sets the closed flag **and** wakes — so a task parked in `next_chunk`/`play` resolves to `Err` instead of hanging forever. `next_chunk` returns `Result<Option<AudioChunk>, _>`: `Ok(Some)` is a chunk, `Ok(None)` is a *graceful* end (a file or mock ran out), and `Err` is device failure — a live mic never ends gracefully, so it reports the failure rather than a `None` indistinguishable from clean exhaustion.

**Lost-wakeup safety.** Both poll paths use register-then-recheck: test the ring, register the waker, test again. Without the second check, a callback firing in the gap between the first check and the registration would leave the task parked forever.

**Layering.** `bridge` (`Signal`, `CaptureRing`, `PlaybackRing`) is **backend-agnostic**: it depends only on `rtrb` + `futures` and speaks `f32`, so any callback-driven backend (a future CoreAudio, AAudio, or JACK one) can reuse it. The cpal-specific glue lives in `source.rs` / `sink.rs` — opening devices, building streams, and the one sample-format-coupled helper `capture_write` (device-native `T` → mono `f32`). Because the bridge owns no device, it is unit-tested by driving the rings directly with a counting waker — no hardware.

**A deliberate simplification.** Waking an `AtomicWaker` from the RT callback is not strictly wait-free (the wake may enqueue the task). At these ~20 ms buffer sizes it is glitch-free in practice; a strictly wait-free bridge is deferred.