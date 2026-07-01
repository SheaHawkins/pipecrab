# PipeCrab
🦀 
Pipecrab is a thoughtful grounds-up rewrite of `pipecat` but in Rust. It aims to be for edge devices what pipecat isn't: A voice agent pipeline for low-latency local inference.

## Writing a stage

A stage implements `Processor`. Both `decide_data` and `decide_system` return a `Decision` — which answers two questions at once: *does the incoming frame keep moving downstream?* and *what should this stage emit?*

| You return | Input frame | Emits |
|---|---|---|
| `Decision::forward()` | forwarded downstream | nothing |
| `Decision::drop()` | consumed | nothing |
| `Decision::drop().emit(x)` | consumed | `x` |
| `Decision::forward().emit(x)` | forwarded downstream | `x` |

**Transform** (e.g. STT, redactor): `drop().emit(output)` — the input never reaches downstream, only the replacement does.

**Tap** (e.g. VAD, logger): `forward().emit(derived)` — the original frame passes through and is followed by the derived one.

**Pass-through**: don't override `decide_data` / `decide_system` — the default is `Decision::forward()`, so every frame on an ignored lane flows on unchanged.

```rust
fn decide_data(&mut self, frame: &DataFrame) -> Decision<Self::Effect> {
    match frame {
        DataFrame::Audio(a) => Decision::drop().emit(Effect::Transcript(self.stt(a))),
        _ => Decision::forward(),
    }
}
```

## Audio I/O

Audio enters and leaves a pipeline through two platform-neutral traits in
[`pipecrab-audio`](./crates/pipecrab-audio): an `AudioSource` (capture) and an
`AudioSink` (playback), both trading in `AudioChunk`s — `f32` PCM samples tagged
with their own `AudioFormat` (sample rate + channels). Chunks ride the pipeline
as the first-party `DataFrame::Audio` variant, so stages match them exhaustively
with no downcast. The crate also ships `mock::MockSource` / `mock::MockSink` for
hardware-free tests.

Concrete backends live behind those traits in their own crates.
[`pipecrab-audio-cpal`](./crates/pipecrab-audio-cpal) is the desktop one
(macOS/Windows/Linux): `CpalSource` / `CpalSink` bridge cpal's real-time device
callbacks to the async pipeline over a lock-free `rtrb` ring buffer, so the
audio thread never blocks, allocates, or locks.

## Running the echo example

[`examples/echo`](./examples/echo) captures your voice and plays it straight
back through a one-stage pipeline — the shortest end-to-end path through the
audio traits, the runtime, and the cpal backend.

```console
$ cargo run -p echo                     # live monitor: hear yourself immediately
$ cargo run -p echo -- --delay-ms 400   # 400 ms delay: an audible echo
$ cargo run -p echo -- --seconds 5      # run for 5 s, then shut down cleanly
```

Use **headphones** — over speakers the mic re-captures the playback and howls.
On macOS the first run triggers a microphone-permission prompt. Without
`--seconds` it runs until Ctrl-C.

## Contributing
See [CONTRIBUTING.md](./CONTRIBUTING.md)
