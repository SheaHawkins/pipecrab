# pipecrab-stt-sherpa

This crate adapts Sherpa ONNX's online and offline recognizers to PipeCrab's
`StreamingTranscriber` protocol. `OnlineSherpaStt` owns a true streaming
`OnlineRecognizer`; `OfflineSherpaStt` accumulates one VAD-bounded utterance
for `OfflineRecognizer` models such as Moonshine v2.

`SherpaStt` and `SherpaSttConfig` are aliases for `OnlineSherpaStt` and
`OnlineSherpaSttConfig`. Callers that want the sensible streaming default use
`SherpaStt::new(config)`.

## Model configuration

### Online streaming transducers

`OnlineSherpaSttConfig::new` takes four streaming-transducer files:

- Encoder ONNX model.
- Decoder ONNX model.
- Joiner ONNX model.
- Token table.

The adapter accepts 16 kHz mono audio and uses CPU greedy search with two
compute threads by default. Sherpa endpoint detection is disabled because the
upstream VAD stage owns utterance boundaries.

Boundary context is model policy rather than a resampling setting. The defaults
are one second before an utterance and 300 milliseconds after it. Both buffers
are allocated once when `SherpaStt` is constructed, and either can be disabled
or tuned per model:

```rust
use std::time::Duration;

let mut config = OnlineSherpaSttConfig::new(encoder, decoder, joiner, tokens);
config.initial_context = Duration::from_millis(750);
config.final_context = Duration::ZERO;
```

The live [stt-sherpa example](../../../examples/stt-sherpa) documents model
downloads and microphone usage.

### Offline Moonshine v2

Moonshine v2 is not available through Sherpa's `OnlineRecognizer`. Its model
layout is an encoder, a merged decoder, and a token table:

```rust
use pipecrab_stt_sherpa::{MoonshineV2Config, OfflineSherpaStt};

let mut config = MoonshineV2Config::new(encoder, merged_decoder, tokens);
config.num_threads = 2;
let transcriber = OfflineSherpaStt::new(config)?;
```

The offline adapter accepts the same 16 kHz mono format and uses CPU greedy
search with two compute threads by default. It buffers audio on its actor
thread between PipeCrab's utterance edges, performs one native decode at the
end, and emits exactly one `Final` event. It cannot emit partial hypotheses.

The live [stt-sherpa-moonshine example](../../../examples/stt-sherpa-moonshine)
documents the official Moonshine v2 model package, microphone usage, and its
VAD-backed integration test.

## Components and ownership

```text
SttStage
   │ StreamingTranscriber
   ▼
OnlineSherpaStt (`SherpaStt` default)
   ├── cancellation generation (atomic)
   └── WorkerHandle
          ├── command sender ───────────────┐
          └── actor thread join handle      │
                                            ▼
                                      worker thread
                                            │ owns
                                            ▼
                                      online backend
                                            │ owns exactly one
                                            ▼
                              sherpa_onnx::OnlineRecognizer
                                            │ creates and outlives
                                            ▼
                                 active OnlineStream (0 or 1)
```

`OnlineSherpaStt` is the inexpensive, `Send + Sync` handle used by `SttStage`.
The actor thread constructs, accesses, and drops the native recognizer. No
recognizer or stream reference escapes that thread.

`WorkerHandle` keeps the command sender and actor join handle together. Dropping
`SherpaStt` closes the command channel and joins the actor. An
active stream is explicitly dropped before the recognizer during worker
shutdown. `OfflineSherpaStt` uses the same handle/actor ownership boundary but
owns an `OfflineRecognizer` and an optional utterance buffer instead of an
`OnlineStream`.

### Backend boundary

`OnlineBackend` (`Backend` is its concise alias) is the testable boundary
around the serialized online-recognizer operations:

```rust
pub trait OnlineBackend: Send + 'static {
    type Stream: 'static;

    fn create_stream(&mut self) -> Self::Stream;
    fn accept_waveform(&mut self, stream: &mut Self::Stream, samples: &[f32]);
    fn input_finished(&mut self, stream: &mut Self::Stream);
    fn is_ready(&mut self, stream: &mut Self::Stream) -> bool;
    fn decode(&mut self, stream: &mut Self::Stream);
    fn get_result(&mut self, stream: &mut Self::Stream) -> Option<String>;
}
```

The production online backend owns one `OnlineRecognizer`. Its associated
stream is `OnlineStream`. Test backends use
ordinary Rust stream values to script hypotheses, block individual decode
steps, and record thread ownership without loading ONNX models.

Mutable receivers express the single-owner rule even where Sherpa's Rust API
accepts shared references internally.

The offline boundary is deliberately utterance-level:

```rust
pub trait OfflineBackend: Send + 'static {
    fn transcribe(&mut self, samples: &[f32]) -> Option<String>;
}
```

Its production implementation creates a fresh `OfflineStream`, accepts the
complete waveform, calls `OfflineRecognizer::decode` once, and reads the final
result. The recognizer remains loaded across utterances.

## Utterance request flow

The private actor protocol has three commands. Every command includes the
cancellation generation observed by its caller and a one-shot reply channel:

```text
Begin { generation, reply }
Feed  { samples: Arc<[f32]>, generation, reply }
End   { generation, reply }
```

Only one utterance can be active per transcriber instance. A second begin, a
feed without begin, or an end without begin becomes `SttError::Engine` rather
than silently changing worker state.

### Begin

`begin_utterance()` creates a clean `OnlineStream`, feeds the configured
zero-valued initial context, and decodes it before accepting utterance audio.
The default is one second. This keeps the recognizer from consuming the first
spoken frames as its startup context. The recognizer remains loaded across
utterances.

### Feed

`feed()` sends the shared `Arc<[f32]>` to the actor without copying its sample
buffer. The production backend calls:

```rust
stream.accept_waveform(16_000, samples);

while recognizer.is_ready(stream) {
    recognizer.decode(stream);
}
```

The actual worker checks cancellation before and after each individual
`decode()` call. It fetches the current result after all ready steps and emits a
partial only when the text changed.

Every partial uses `stable: 0`. Sherpa supplies a current best hypothesis but
does not guarantee that any prefix is permanent, so the adapter does not claim
stability from a longest-common-prefix heuristic. A hypothesis can retract all
the way to an empty string.

### End

`end_utterance()` appends the configured zero-valued final context before
calling `input_finished()`. The default is 300 milliseconds. Streaming
Zipformer needs that right context to finish tokens near the end of a short
utterance. The worker then drains every
remaining ready decode step, reads the final result, and drops the stream. It
always returns exactly one `SttEvent::Final`, including an empty final when
Sherpa recognizes no text.

The final padding follows Sherpa's
[official Rust streaming Zipformer example](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/streaming_zipformer.rs),
which adds approximately 0.3 seconds of silence before finishing a stream.

Padding supplies encoder context; it does not guarantee that every short sound
becomes a token. In particular, the recommended 20M model can return blank for
an isolated one-syllable word even when the same waveform bypasses VAD. Treat
reliable keyword recognition as a model-selection or contextual-biasing
requirement rather than increasing VAD pre-roll indefinitely.

PipeCrab's `SpeechStopped` edge is the boundary authority. Sherpa endpoint
detection is disabled, and the worker never calls `is_endpoint()`.

### Offline flow

`OfflineSherpaStt::begin_utterance()` creates an empty actor-owned sample
buffer. Each `feed()` appends its chunk and returns no events.
`end_utterance()` moves the completed buffer into the offline backend and
returns one `SttEvent::Final`, including an empty final when Sherpa recognizes
no text. There is no extra recognizer padding; VAD pre-roll and trailing
silence are already part of the bounded waveform.

## Cancellation and interruption

`StreamingTranscriber::cancel()` is synchronous, non-blocking, idempotent, and
infallible. It only increments an `AtomicU64`; it does not enqueue work or wait
for the actor.

The worker checks command generations:

- Before touching a queued command.
- Before every decode step.
- After every decode step.
- Before returning a hypothesis or final result.

The public handle checks the generation again after receiving the actor reply.
Together these checks prevent stale output in four races:

1. Cancellation before the actor reads a queued command.
2. Cancellation between ready decode steps.
3. Cancellation during a native decode call.
4. Cancellation after reply delivery but before the awaiting caller resumes.

When the worker observes a new generation, it drops the active stream and
clears its remembered partial. Commands from older generations are answered
without touching recognizer state, so they cannot contaminate the next
utterance.

Sherpa does not expose cancellation inside one already-running `decode()` call.
For `OnlineSherpaStt`, that native call completes before the worker checks the
generation again; small chunks and one decode step per check bound the delay.
For `OfflineSherpaStt`, the single utterance-level decode must finish before the
actor can observe cancellation, but both the actor and awaiting caller discard
its stale result. A one- or two-thread inference pool limits contention, not
the duration of an already-running offline call.

## Composition with Sherpa VAD

`pipecrab-vad-sherpa` and this crate use the same pinned `sherpa-onnx` package,
but they do not share mutable native objects or actor threads:

```text
continuous 16 kHz audio
          │
          ▼
SherpaVad actor
  one VoiceActivityDetector
  one compute thread
          │ speech edges + gated Arc-backed chunks
          ▼
Sherpa STT actor
  one OnlineRecognizer + active stream
  or one OfflineRecognizer + utterance buffer
  one or two compute threads
```

`VadStage` owns pre-roll and forwards the original shared audio chunks between
`SpeechStarted` and `SpeechStopped`. `SttStage` translates those frames directly
to begin, feed, and end commands. Both adapters declare 16 kHz mono input, so a
single resampler belongs before VAD.

The microphone example uses a transcription-oriented VAD profile:

- 0.35 speech threshold.
- 100 ms minimum speech duration.
- One second of gate pre-roll.
- 500 ms minimum trailing silence.
- 30-second maximum speech duration.

The longer pre-roll preserves speech that occurred before Silero confirmed the
onset. The lower threshold and shorter confirmation window admit brief words
such as “hi.” The silence hangover reduces utterance fragmentation, while the
larger duration ceiling avoids forcing ordinary spoken turns closed every five
seconds.

The actor split is intentional. STT decoding is substantially heavier than one
Silero window and must not block VAD's command queue. Workers are not pinned to
specific CPU cores.

## Errors

`SherpaSttBuildError` separates invalid paths or thread counts, online or
offline recognizer construction failure, and worker setup failure. Runtime
channel closure, protocol violations, and actor failure become
`SttError::Engine` so `SttStage` can surface them through the pipeline's normal
recoverable-error path.

## Model-backed integration tests

Default `cargo test` runs the deterministic backend tests and compiles the
model-backed tests, but does not execute them. The model-backed tests are
`#[ignore]` because the repository does not vendor the ONNX files.

The speech input is committed at
[`test-resources/audio/sherpa-zipformer-en-20m-0.wav`](../../../test-resources/audio/sherpa-zipformer-en-20m-0.wav),
so no WAV environment variable is required.

The Moonshine tests use the same committed resources. No WAV should be copied
from a downloaded model directory or referenced through an environment
variable.

Run these commands from the repository root. The environment variables use
absolute paths because Cargo runs integration-test binaries with a package
working directory.

### Direct streaming STT

This test feeds the committed WAV to `SherpaStt` in 512-sample chunks and
requires the final transcript to retain its opening words:

```console
ROOT="$(pwd)"
MODEL="$ROOT/models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

SHERPA_STT_ENCODER="$MODEL/encoder-epoch-99-avg-1.int8.onnx" \
SHERPA_STT_DECODER="$MODEL/decoder-epoch-99-avg-1.onnx" \
SHERPA_STT_JOINER="$MODEL/joiner-epoch-99-avg-1.int8.onnx" \
SHERPA_STT_TOKENS="$MODEL/tokens.txt" \
cargo test -p pipecrab-stt-sherpa \
  --test transcriber \
  production_backend_transcribes_a_wave_file \
  -- --ignored --nocapture
```

`--nocapture` prints the recognized final text.

### VAD-gated streaming STT

This test exercises the same topology as the microphone example:

```text
48 kHz WAV → ResamplerStage → SherpaVad → VadStage → SherpaStt → SttStage
```

It appends one second of silence to close the VAD utterance and requires the
final transcript to retain the fixture's opening words:

```console
ROOT="$(pwd)"
MODEL="$ROOT/models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

SHERPA_VAD_MODEL="$ROOT/models/silero_vad.onnx" \
SHERPA_STT_ENCODER="$MODEL/encoder-epoch-99-avg-1.int8.onnx" \
SHERPA_STT_DECODER="$MODEL/decoder-epoch-99-avg-1.onnx" \
SHERPA_STT_JOINER="$MODEL/joiner-epoch-99-avg-1.int8.onnx" \
SHERPA_STT_TOKENS="$MODEL/tokens.txt" \
cargo test -p stt-sherpa \
  --test model_pipeline \
  vad_gated_wave_produces_a_final_transcript \
  -- --ignored --nocapture
```

The short fixture contains about 400 milliseconds of speech:

```console
SHERPA_VAD_MODEL="$ROOT/models/silero_vad.onnx" \
SHERPA_STT_ENCODER="$MODEL/encoder-epoch-99-avg-1.int8.onnx" \
SHERPA_STT_DECODER="$MODEL/decoder-epoch-99-avg-1.onnx" \
SHERPA_STT_JOINER="$MODEL/joiner-epoch-99-avg-1.int8.onnx" \
SHERPA_STT_TOKENS="$MODEL/tokens.txt" \
cargo test -p stt-sherpa \
  --test model_pipeline \
  vad_gated_short_wave_produces_text \
  -- --ignored --nocapture
```

To isolate recognizer startup from VAD, run the 48→16 kHz resampling path
directly into `SherpaStt`:

```console
ROOT="$(pwd)"
MODEL="$ROOT/models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

SHERPA_STT_ENCODER="$MODEL/encoder-epoch-99-avg-1.int8.onnx" \
SHERPA_STT_DECODER="$MODEL/decoder-epoch-99-avg-1.onnx" \
SHERPA_STT_JOINER="$MODEL/joiner-epoch-99-avg-1.int8.onnx" \
SHERPA_STT_TOKENS="$MODEL/tokens.txt" \
cargo test -p stt-sherpa \
  --test model_pipeline \
  transcribes_microphone_resampling_without_vad \
  -- --ignored --nocapture
```

### Direct Moonshine v2 STT

This test feeds the committed 16 kHz WAV through `OfflineSherpaStt`:

```console
ROOT="$(pwd)"
MODEL="$ROOT/models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27"

SHERPA_MOONSHINE_ENCODER="$MODEL/encoder_model.ort" \
SHERPA_MOONSHINE_MERGED_DECODER="$MODEL/decoder_model_merged.ort" \
SHERPA_MOONSHINE_TOKENS="$MODEL/tokens.txt" \
cargo test -p pipecrab-stt-sherpa \
  --test offline_transcriber \
  moonshine_v2_transcribes_a_wave_file \
  -- --ignored --nocapture
```

### VAD-gated Moonshine v2 STT

This test exercises the microphone topology with a committed 48 kHz WAV:

```console
ROOT="$(pwd)"
MODEL="$ROOT/models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27"

SHERPA_VAD_MODEL="$ROOT/models/silero_vad.onnx" \
SHERPA_MOONSHINE_ENCODER="$MODEL/encoder_model.ort" \
SHERPA_MOONSHINE_MERGED_DECODER="$MODEL/decoder_model_merged.ort" \
SHERPA_MOONSHINE_TOKENS="$MODEL/tokens.txt" \
cargo test -p stt-sherpa-moonshine \
  --test model_pipeline \
  -- --ignored --nocapture
```

## Default CI coverage

CI does not need model files or network access. It covers:

- Configuration validation and fixed Sherpa settings.
- Partial and final transcript behavior.
- Offline buffering and final-only transcript behavior.
- Duplicate-hypothesis suppression.
- Protocol errors.
- Actor-thread ownership.
- Cancellation during native decoding.
- Stale offline-result suppression after cancellation.
- Stale queued-command rejection.
- Worker failure reporting.
- Resampled waveform timing, level, and similarity.
- Compilation of all ignored model-backed tests.

To run the default crate suite locally:

```console
cargo test -p pipecrab-stt-sherpa
cargo test -p stt-sherpa
cargo test -p stt-sherpa-moonshine
```
