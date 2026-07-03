We differ from pipecat in a few ways:

1. Pipecat Processor = Pipecrab Stage
1. The pipeline is split into two lanes: Data and Sys (high-priority). Messages in the sys queue are drained first and can interrupt work in progress, as well as travel to stage upstream (such as for configuration requests). The data lane is one-directional and flows downstream only.
1. A Stage can manage internal state via the uninterruptable synchronous `decide` function. By contrast, the async `perform` function can be interrupted but cannot modify state. This prevents broken state 
1. SystemFrames are distinct from DataFrames. In pipecrab, they don't share an inheritance tree so you can't accidentally push SystemFrames into the downstream-only data lane. 
1. We are explicit about whether Stages `forward` or `drop` frames. It's enforced by the compiler so you can't [accidentally forget to push frames](https://docs.pipecat.ai/pipecat/fundamentals/custom-frame-processor#example-metricsframe-logger).
1. Pipecat treats InputAudio as a SystemFrame. That's a wart here. There's a system lane for `Interrupt` and Audio rides the data lane with flush resistance. When an interrupt comes through, it flushes non-survivor frames (i.e., flushes everything other than Audio).

## Crates

Dependencies point downward only — backend → trait crate → core.

- `pipecrab-core` — sans-IO: frames, `Processor`, `Decision`. No async, no I/O.
- `pipecrab-runtime` — async orchestration: `Stage`, `Pipeline`, `Inbound`/`Outbound`, `offload`. No executor baked in.
- `pipecrab` — facade; re-exports core + runtime.
- `pipecrab-audio` — `AudioSource`/`AudioSink` traits + hardware-free mocks.
- `pipecrab-audio-cpal` — cpal backend behind those traits.
- `pipecrab-stt` — `Transcriber` trait + `SttStage` adapter.

## Off-thread work

The pipeline is one `!Send` thread, and `perform` must keep yielding so an interrupt can preempt it. Heavy work never runs inline: `offload(f).await` crosses to a `std::thread` (native) or Web Worker (wasm). It is the only API with a `Send` bound — that bound is the thread-crossing boundary, not the pipeline.

## Runs on wasm

Every async trait is `?Send` and no executor is baked in, so one stage definition runs on a current-thread executor and in the browser; the caller drives the future (`block_on` natively, `spawn_local` in the browser). CI compiles core, the runtime, and every trait crate for `wasm32-unknown-unknown`, so a host-only dependency can't creep in unnoticed.

## Audio bridge

cpal's device callbacks run on a real-time thread that must never block, allocate, or lock; the pipeline is async. `pipecrab-audio-cpal::bridge` reconciles them with one lock-free `rtrb` ring per direction (moves `f32` samples) plus a `Signal` — an `AtomicWaker` and a "stream died" flag (moves wakeups). Capture: the RT callback produces, `CaptureRing` consumes; a full ring drops samples as an *overrun*. Playback: `PlaybackRing` produces with backpressure, the RT callback consumes; an empty ring outputs silence. `Signal::fail` wakes a parked task to `Err` on device loss. The bridge touches no cpal — only `rtrb` + `futures` — so it is backend-agnostic and tested without hardware.

## STT seam

`Transcriber` (`f32` in, text out) is the swappable capability; `SttStage` adapts one to a `Stage`, replacing a `DataFrame::Audio` with a `DataFrame::Transcript`. Engines live behind the trait — native `ort`, browser Transformers.js in a Worker — each owning its own offload, so the pipeline never names a concrete model.
