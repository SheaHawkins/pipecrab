# pipecrab-tts-sherpa

This crate adapts Sherpa ONNX's offline text-to-speech engine to PipeCrab's
`Synthesizer` protocol. `SherpaTts` owns the native engine on a dedicated
actor thread; `KokoroConfig` names the Kokoro model files.

## Model configuration

`KokoroConfig::new` takes the four artifacts every Kokoro package ships:

- The Kokoro ONNX model.
- The packed voice-embedding file (`voices.bin`).
- The token table.
- The espeak-ng data directory.

The adapter uses the CPU provider with two compute threads by default. The
engine reports its own output format (24 kHz mono for the published Kokoro
models) through `Synthesizer::output_format`; a resampler stage downstream
converts to the playback rate.

```rust
use pipecrab_tts_sherpa::{KokoroConfig, SherpaTts};

let mut config = KokoroConfig::new(model, voices, tokens, data_dir);
config.speaker = 8;   // one of the model's built-in voices
config.speed = 1.2;   // speaking-rate multiplier
let synth = SherpaTts::new(config)?;
```

The multi-language Kokoro packages add optional fields: `dict_dir` (jieba
dictionary), `lexicon` (comma-separated lexicon files), and `lang`.

The live [e2e-voice-agent example](../../../examples/e2e-voice-agent)
documents model downloads and the full voice-agent pipeline.

## Components and ownership

```text
TtsStage
   │ Synthesizer
   ▼
SherpaTts
   ├── cancellation epoch (atomic)
   ├── cached output AudioFormat
   └── WorkerHandle
          ├── command sender ───────────────┐
          └── actor thread join handle      │
                                            ▼
                                      worker thread
                                            │ owns
                                            ▼
                                      Kokoro backend
                                            │ owns exactly one
                                            ▼
                                  sherpa_onnx::OfflineTts
```

`SherpaTts` is the inexpensive, `Send + Sync` handle used by `TtsStage`. The
actor thread constructs, accesses, and drops the native engine; no engine
reference escapes it. Model loading runs on the actor during construction, and
`SherpaTts::new` waits for setup so a returned handle is ready and already
knows the model's output format. Dropping `SherpaTts` bumps the epoch (so an
in-flight generation stops within one sentence), closes the command channel,
and joins the actor.

### Backend boundary

`Backend` is the testable boundary around the serialized synthesis operation:

```rust
pub trait Backend: Send + 'static {
    fn sample_rate(&mut self) -> u32;
    fn generate(&mut self, text: &str, emit: Emit) -> Result<(), String>;
}
```

`Emit` is a boxed `FnMut(&[f32]) -> bool`; boxed because Sherpa's native
callback requires an owned `'static` value. The production backend calls
`OfflineTts::generate_with_config` and forwards each callback invocation to
`emit`. Test backends script segment sequences, block mid-generation, and
record thread ownership without loading ONNX models.

## Synthesis flow

`synthesize(text)` takes a fresh epoch, sends one command to the actor, and
returns an unbounded receiver boxed as the `TtsAudioStream`. Sherpa generates
offline audio *sentence by sentence* and invokes its progress callback with
each newly generated sentence's samples; the worker wraps each in an
`AudioChunk` carrying the engine's format and sends it to the stream
immediately. Playback of a long reply therefore starts after its first
sentence, upstream of any `SentenceChunker` splitting.

When generation finishes (or fails — the error is delivered on the stream as
`TtsError::Engine`), the worker drops its sender and the stream ends.

## Cancellation and interruption

`Synthesizer::cancel()` is synchronous, non-blocking, idempotent, and
infallible: it only increments an `AtomicU64`.

The engine observes it through the progress callback's return value. The
worker's callback returns `false` — telling Sherpa to stop generating — when
either:

1. The epoch has advanced (a barge-in `cancel()`, a newer `synthesize`, or
   the handle dropping), or
2. The stream's receiver was dropped (the stage stopped pulling).

Sherpa checks between sentences, so cancellation latency is bounded by one
sentence of synthesis. Stale commands (an epoch that advanced before the actor
dequeued them) are skipped without touching the engine, and a stale
generation's error is never delivered.

## Errors

`SherpaTtsBuildError` separates invalid paths or parameters, engine
construction failure, and worker setup failure. Runtime failures — a failed
generation, a stopped worker — become `TtsError::Engine`, which `TtsStage`
surfaces through the pipeline's normal recoverable-error path (the utterance
is skipped; the pipeline lives on).

## Model-backed integration test

Default `cargo test` runs the deterministic backend tests and compiles the
model-backed test, but does not execute it, because the repository does not
vendor the model files.

Run from the repository root with absolute paths (Cargo runs integration-test
binaries with a package working directory):

```console
TTS="$PWD/models/kokoro-en-v0_19"

SHERPA_KOKORO_MODEL="$TTS/model.onnx" \
SHERPA_KOKORO_VOICES="$TTS/voices.bin" \
SHERPA_KOKORO_TOKENS="$TTS/tokens.txt" \
SHERPA_KOKORO_DATA_DIR="$TTS/espeak-ng-data" \
cargo test -p pipecrab-tts-sherpa --test synthesizer -- --ignored --nocapture
```

## Default CI coverage

CI does not need model files or network access. It covers:

- Configuration validation and fixed Sherpa settings.
- Segment streaming with the engine-reported format.
- Empty-segment suppression.
- Consecutive syntheses on one handle.
- Actor-thread ownership.
- Cancellation stopping an in-flight generation.
- A dropped stream stopping the engine.
- Engine-error delivery on the stream.
- Worker setup failure for a zero sample rate.
- Compilation of the ignored model-backed test.

To run the default crate suite locally:

```console
cargo test -p pipecrab-tts-sherpa
```
