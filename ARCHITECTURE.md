We differ from pipecat in a few ways:

1. Pipecat Processor = Pipecrab Stage
1. The pipeline is split into two lanes: Data and Sys (high-priority). Messages in the sys queue are drained first and can interrupt work in progress, as well as travel to stage upstream (such as for configuration requests). The data lane is one-directional and flows downstream only.
1. The system lane is only for frames that are allowed to overtake — **Interrupt, Start/Stop, Error**. Do NOT use the system lane for state changes (SpeechStarted/Stopped). 
1. A Stage can manage internal state via the uninterruptable synchronous `decide` function. By contrast, the async `perform` function can be interrupted but cannot modify state. This prevents broken state
1. SystemFrames are distinct from DataFrames. In pipecrab, they don't share an inheritance tree so you can't accidentally push SystemFrames into the downstream-only data lane.
1. We are explicit about whether Stages `forward` or `drop` frames. It's enforced by the compiler so you can't [accidentally forget to push frames](https://docs.pipecat.ai/pipecat/fundamentals/custom-frame-processor#example-metricsframe-logger).
1. Pipecat treats InputAudio as a SystemFrame. That's a wart here. There's a system lane for `Interrupt` and Audio rides the data lane with flush resistance. When an interrupt comes through, it flushes non-survivor frames (input audio and durable model/dispatch frames survive).

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

The synchronous `decide_*` half owns state mutation; the async `perform` half does the emitting I/O but cannot mutate state (see "Off-thread work"). To compose stages into a runnable pipeline, see [Writing a pipeline](./README.md#writing-a-pipeline).

## Data-lane frame vocabulary

Two native protocol families ride the data lane alongside audio, transcripts, and voice edges:

- **`Model(ModelFrame)`** — one LM generation: `GenerationStarted`/`GenerationFinished`, `ToolCall`, `Input` adds non-user `Context`/`Respond` messages.
- **`Dispatch(DispatchFrame)`** — async tasks: a `Command` drives one, an `Event` reports state. `tool_call_id` names the invocation; `task_id` (post-`Accepted`) the task.

They use the data lane, not the system lane, because order matters. Text may precede *or* follow a tool call. On interrupt, `survives_flush` keeps `InputAudio`, `Model(Input)`, `Model(ToolCall)`, and every `Dispatch`.

## Crates

Crates are grouped by role under `crates/`, and dependencies point downward
only — adapter → trait crate → runtime → core:

- `crates/engine/*` — the pipecrab framework (core, runtime, the facade, and
  the capability trait crates).
- `crates/adapters/*` — crates that bridge an external *engine* (a model
  runtime like sherpa-onnx) or a device (cpal) to a capability trait.
- `crates/support/*` — dev-only tooling; never published (`pipecrab-arch`
  holds the layering gate; `pipecrab-test-util` holds shared test helpers).

The folder grouping is cosmetic to Cargo — it resolves deps by name — so the
downward-only rule is enforced, not merely suggested, by the `layering` test in
`crates/support/arch` (see "Layering gate" below). Note "engine" is overloaded:
the `engine/` **folder** is the framework, whereas an "engine" in prose is the
external model runtime an adapter wraps.

- `pipecrab-core` — sans-IO: frames, `Processor`, `Decision`. No async, no I/O.
- `pipecrab-runtime` — async orchestration: `Stage`, `Pipeline`, `Inbound`/`Outbound`, `offload`. No executor baked in.
- `pipecrab` — facade; re-exports core + runtime.
- `pipecrab-audio` — `AudioSource`/`AudioSink` traits, hardware-free mocks, and
  the streaming `ResamplerStage`.
- `pipecrab-audio-cpal` — cpal backend behind those traits.
- `pipecrab-stt` — `Transcriber` trait + `SttStage` adapter.
- `pipecrab-vad` — two-tier VAD: the `VoiceActivityDetector` trait (audio in, speech edges out) that `VadStage` gates on, plus the `SpeechScorer` raw-model tier and the `Debounced` adapter that lifts a scorer into a detector. See "VAD gate" below.
- `pipecrab-stt-sherpa`, `pipecrab-vad-sherpa` — adapter crates: implement the corresponding traits by wrapping an external engine (`sherpa-onnx`). These live under `crates/adapters/` alongside `pipecrab-audio-cpal`.

## Layering gate

The dependency direction above is an enforced invariant, not a convention.
Every crate declares its layer in its own manifest:

```toml
[package.metadata.pipecrab]
layer = "runtime"   # core < runtime < {trait, facade} < adapter < app
```

`crates/support/arch/tests/layering.rs` reads the resolved package graph
(`cargo metadata`) and fails `cargo test --workspace` if any crate's normal or
build dependency points to an equal-or-higher layer. Two properties make it
low-maintenance:

- **Layer is declared, not inferred from the folder.** Moving a crate between
  folders changes nothing; the manifest is the source of truth, so the fine
  ordering (core < runtime < trait) survives the coarse two-folder split.
- **It fails closed.** A workspace member with no declared layer is an error, so
  a new crate can't slip through unlabeled — it forces a one-line decision at
  creation. `support` is a valid layer that opts a dev-only crate out of the
  ordering.

Dev-dependencies are exempt (a test may reach for anything). The acyclic
guarantee Cargo already provides is a partial backstop — an engine crate
depending on an adapter that routes back to it is a hard cycle error — but the
gate covers the acyclic cases Cargo permits.

## Crating strategy

```
you depend on ──────────────────────────────────────────────────────────
  your-app/          graph.rs (shared) + main.rs / web.rs (thin roots)
  ├── pipecrab                       umbrella: core + runtime re-exports
  ├── pipecrab-stt-sherpa            model crates (one per capability):
  ├── pipecrab-vad-sherpa            all policy, public Backend trait,
  │                                  re-export everything you need
  └── pipecrab-audio-cpal            audio edge for your platform

pulled in for you ──────────────────────────────────────────────────────
  sherpa-onnx                        engine: pipecrab-free, cfg-selected
      ▲ wrapped by model crates          
  pipecrab-stt · pipecrab-vad · pipecrab-audio   trait crates: capability
      ▲ implemented by model crates               trait + Stage adapter
  pipecrab-runtime   Stage, two lanes, run loop; Timer/Offload definitions
      ▲
  pipecrab-core      frames, Processor — zero deps, no async, no cfg
```

## Off-thread work

The pipeline is one `!Send` thread, and `perform` must keep yielding so an interrupt can preempt it. Heavy work never runs inline: `offload(f).await` crosses to a `std::thread` (native) or Web Worker (wasm). It is the only API with a `Send` bound — that bound is the thread-crossing boundary, not the pipeline.

## Runs on wasm

Every async trait carries a target-conditional `Send` bound (`MaybeSend`/`MaybeSendSync`) — real on native so a multi-threaded executor can migrate the task, vacuous on `wasm32` where `Send` can't hold — and no executor is baked in, so one stage definition runs on a current-thread executor and in the browser; the caller drives the future (`block_on` natively, `spawn_local` in the browser). Where a native handle is `!Send` (cpal's `Stream`), the backend keeps it off the struct rather than relaxing the bound. CI compiles core, the runtime, and every trait crate for `wasm32-unknown-unknown`, so a host-only dependency can't creep in unnoticed.

## Audio bridge

cpal's device callbacks run on a real-time thread that must never block, allocate, or lock; the pipeline is async. `pipecrab-audio-cpal::bridge` reconciles them with one lock-free `rtrb` ring per direction (moves `f32` samples) plus a `Signal` — an `AtomicWaker` and a "stream died" flag (moves wakeups). Capture: the RT callback produces, `CaptureRing` consumes; a full ring drops samples as an *overrun*. Playback: `PlaybackRing` produces with backpressure, the RT callback consumes; an empty ring outputs silence. `Signal::fail` wakes a parked task to `Err` on device loss. The bridge touches no cpal — only `rtrb` + `futures` — so it is backend-agnostic and tested without hardware. A `cpal::Stream` is `!Send`, but the interface is `Send`; so the stream is built and parked on a dedicated owning thread and `CpalSource`/`CpalSink` hold only the `Send` ring end (a server spawns one pump per session). This cpal backend is native-only — the browser audio path will be a separate crate.

## STT interface

`Transcriber` (`f32` in, text out) is the swappable capability; `SttStage` adapts one to a `Stage`, replacing a `DataFrame::Audio` with a `DataFrame::Transcript`. Audio crosses the async engine boundary as `Arc<[f32]>`, allowing worker-backed engines to retain or enqueue a chunk without copying its samples. Engines live behind the trait — native `ort`, browser Transformers.js in a Worker — each owning its own offload, so the pipeline never names a concrete model.

## VAD gate

VAD is two tiers. `VoiceActivityDetector` (audio in, speech *edges* out) is the stage-facing capability that segmenter-class engines (sherpa's VAD, platform VADs) implement directly. `SpeechScorer` (a per-window probability) is the raw-model tier a bare silero build exposes; `Debounced` lifts a scorer into a detector, owning the windowing, threshold, and hangover. Because a segmenter never passes through `Debounced`, no engine is ever debounced twice.

`VadStage` is a **gate**, not a tap. It runs the detector over every conforming chunk and emits, on the data lane, an utterance's audio **bracketed by its edges** — `SpeechStarted`, then the utterance's chunks (pre-roll included, drained from a ring the gate owns), then `SpeechStopped`. While idle it emits nothing.

**Contract inversion.** The older lane-discipline design made VAD a tap and emitted the edge *after* the chunk that triggered it. The gate inverts this: downstream of `VadStage`, edges bracket the utterance audio — `SpeechStarted` precedes every utterance chunk, `SpeechStopped` follows the last. A downstream stage can therefore be stateless off the edges, opening its utterance on `SpeechStarted` alone.

**Topology commitment.** Because the gate drops silence, any future consumer of *continuous* raw audio (recording, a level meter, an AEC reference) must sit **upstream** of `VadStage`. Nothing downstream consumes silence today, so this constrains nothing yet — but it is a standing commitment.

**Format is fatal.** `Arc<[f32]>` carries no sample rate, so the detector declares its `input_format()` and the stage enforces it fatally: nonconforming audio is rejected before it reaches any engine (`VadError` therefore carries no format-mismatch variant). The shared buffer lets worker-backed detectors retain a chunk without copying its samples. Continuous-format conversion belongs to a resample stage upstream.

## Resampling

`pipecrab-audio::Resampler` is the synchronous audio-to-audio conversion
interface. `ResamplerStage<R>` adapts any implementation to the pipeline;
`ResamplerStage::new(format)` selects the bundled `RubatoSincResampler` as the
sensible default, while `with_resampler` accepts another backend. Non-audio
frames pass through unchanged.

The Rubato sinc implementation is continuous across chunks of one input
format. An input-format change or `Interrupt` resets its filter history. Equal
channel counts remain independent; when counts differ, channels are averaged
to mono and replicated because `AudioFormat` carries no speaker-layout metadata
from which to infer a more specific mix matrix.

The DSP state lives in `decide_data`, keeping mutation inside the synchronous,
non-cancellable half of the stage. That work occupies the orchestrator for the
duration of one input chunk, so the implementation uses reusable buffers and a
small fixed internal block. The `pipecrab-audio` resampler benchmark asserts
cold-start and steady-state occupancy as a fraction of represented audio time.
