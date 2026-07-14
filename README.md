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
[`pipecrab-audio`](./crates/engine/pipecrab-audio): an `AudioSource` (capture) and an
`AudioSink` (playback), both trading in `AudioChunk`s — `f32` PCM samples tagged
with their own `AudioFormat` (sample rate + channels). Chunks ride the pipeline
as the first-party `DataFrame::Audio` variant, so stages match them exhaustively
with no downcast. The crate also ships `mock::MockSource` / `mock::MockSink` for
hardware-free tests.

Concrete backends live behind those traits in their own crates.
[`pipecrab-audio-cpal`](./crates/adapters/pipecrab-audio-cpal) is the cpal one — it runs
wherever cpal does (macOS, Windows, Linux, iOS, Android, and the browser via
WebAudio). `CpalSource` / `CpalSink` bridge cpal's real-time device callbacks to
the async pipeline over a lock-free `rtrb` ring buffer, so the audio thread does
no locking or allocation on the hot path — it only hands samples across the ring
and wakes the async side (that wake is a documented pragmatic exception). Both
open from a shared `CpalConfig` (which device per side, plus chunk/buffer
sizing), defaulting to the system default devices.

`pipecrab-audio` also provides the `Resampler` audio-to-audio interface and a
generic `ResamplerStage<R>` pipeline adapter. `ResamplerStage::new(format)` uses
the bundled `RubatoSincResampler`; `ResamplerStage::with_resampler(custom)`
injects another implementation. Put the stage immediately before a
format-strict VAD, STT engine, or audio sink.

[`pipecrab-vad-sherpa`](./crates/adapters/pipecrab-vad-sherpa) runs Sherpa
ONNX's Silero VAD on a dedicated worker and implements the edge-emitting
`VoiceActivityDetector` interface. It accepts 16 kHz mono audio in exact
512-sample model windows; arbitrary PipeCrab chunk sizes are accumulated by the
adapter.

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

## Running the [Sherpa VAD example](./examples/vad-sherpa)

Download Sherpa's 16 kHz Silero model, then run the live microphone example:

```console
$ curl -L \
    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
    -o silero_vad.onnx
$ cargo run -p vad-sherpa -- --model ./silero_vad.onnx
$ cargo run -p vad-sherpa -- --model ./silero_vad.onnx --seconds 10
```

The example captures at the microphone's native rate, resamples once to 16 kHz
mono, and prints `SpeechStarted` and `SpeechStopped`. The first Sherpa build may
download matching native libraries; set `SHERPA_ONNX_LIB_DIR` to use an existing
Sherpa installation.

## Contributing
See [CONTRIBUTING.md](./CONTRIBUTING.md)
