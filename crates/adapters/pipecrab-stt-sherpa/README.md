# pipecrab-stt-sherpa

This crate adapts Sherpa ONNX's online recognizer to PipeCrab's
`StreamingTranscriber` protocol. One actor thread owns the
`OnlineRecognizer` and its active `OnlineStream`; PipeCrab VAD edges create and
finish each utterance.

## Model configuration

`SherpaSttConfig::new` takes four streaming-transducer files:

- Encoder ONNX model.
- Decoder ONNX model.
- Joiner ONNX model.
- Token table.

The adapter accepts 16 kHz mono audio and uses CPU greedy search with two
compute threads by default. Sherpa endpoint detection is disabled because the
upstream VAD stage owns utterance boundaries.

The live [stt-sherpa example](../../../examples/stt-sherpa) documents model
downloads and microphone usage.

## Components and ownership

```text
SttStage
   │ StreamingTranscriber
   ▼
SherpaStt
   ├── cancellation generation (atomic)
   └── WorkerHandle
          ├── command sender ───────────────┐
          └── actor thread join handle      │
                                            ▼
                                      worker thread
                                            │ owns
                                            ▼
                                      SherpaBackend
                                            │ owns exactly one
                                            ▼
                              sherpa_onnx::OnlineRecognizer
                                            │ creates and outlives
                                            ▼
                                 active OnlineStream (0 or 1)
```

`SherpaStt` is the inexpensive, `Send + Sync` handle used by `SttStage`. The
actor thread constructs, accesses, and drops the native recognizer. No
recognizer or stream reference escapes that thread.

`WorkerHandle` keeps the command sender and actor join handle together. Dropping
`SherpaStt` closes the command channel and joins the actor. An
active stream is explicitly dropped before the recognizer during worker
shutdown.

### Backend boundary

`Backend` is the testable boundary around the serialized online-recognizer
operations:

```rust
pub trait Backend: Send + 'static {
    type Stream: 'static;

    fn create_stream(&mut self) -> Self::Stream;
    fn accept_waveform(&mut self, stream: &mut Self::Stream, samples: &[f32]);
    fn input_finished(&mut self, stream: &mut Self::Stream);
    fn is_ready(&mut self, stream: &mut Self::Stream) -> bool;
    fn decode(&mut self, stream: &mut Self::Stream);
    fn get_result(&mut self, stream: &mut Self::Stream) -> Option<String>;
}
```

`SherpaBackend` is the production implementation and owns one
`OnlineRecognizer`. Its associated stream is `OnlineStream`. Test backends use
ordinary Rust stream values to script hypotheses, block individual decode
steps, and record thread ownership without loading ONNX models.

Mutable receivers express the single-owner rule even where Sherpa's Rust API
accepts shared references internally.

## Utterance request flow

The private actor protocol has three commands. Every command includes the
cancellation generation observed by its caller and a one-shot reply channel:

```text
Begin { generation, reply }
Feed  { samples: Arc<[f32]>, generation, reply }
End   { generation, reply }
```

Only one utterance can be active per `SherpaStt` instance. A second begin, a
feed without begin, or an end without begin becomes `SttError::Engine` rather
than silently changing worker state.

### Begin

`begin_utterance()` creates a clean `OnlineStream` and clears the previous
partial hypothesis. The recognizer remains loaded across utterances.

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

`end_utterance()` appends 300 milliseconds of zero-valued audio before calling
`input_finished()`. Streaming Zipformer needs that right context to finish
tokens near either edge of a short utterance. The worker then drains every
remaining ready decode step, reads the final result, and drops the stream. It
always returns exactly one `SttEvent::Final`, including an empty final when
Sherpa recognizes no text.

PipeCrab's `SpeechStopped` edge is the boundary authority. Sherpa endpoint
detection is disabled, and the worker never calls `is_endpoint()`.

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
That native call completes before the worker can observe the new generation;
its result is then discarded. Small input chunks, one decode step per check,
and a one- or two-thread inference pool bound that delay.

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
SherpaStt actor
  one OnlineRecognizer
  one active OnlineStream
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

`SherpaSttBuildError` separates invalid paths or thread counts, recognizer
construction failure, and worker setup failure. Runtime channel closure,
protocol violations, and actor failure become `SttError::Engine` so `SttStage`
can surface them through the pipeline's normal recoverable-error path.

## Model-backed integration tests

Default `cargo test` runs the deterministic backend tests and compiles the
model-backed tests, but does not execute them. The model-backed tests are
`#[ignore]` because the repository does not vendor the ONNX files.

The speech input is committed at
[`test-resources/audio/sherpa-zipformer-en-20m-0.wav`](../../../test-resources/audio/sherpa-zipformer-en-20m-0.wav),
so no WAV environment variable is required.

Run these commands from the repository root. The environment variables use
absolute paths because Cargo runs integration-test binaries with a package
working directory.

### Direct streaming STT

This test feeds the committed WAV to `SherpaStt` in 512-sample chunks and
requires a non-empty final transcript:

```console
ROOT="$(pwd)"
MODEL="$ROOT/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

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
known WAV → SherpaVad → VadStage → SherpaStt → SttStage → final transcript
```

It appends one second of silence to close the VAD utterance and requires a
non-empty final transcript:

```console
ROOT="$(pwd)"
MODEL="$ROOT/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

SHERPA_VAD_MODEL="$ROOT/silero_vad.onnx" \
SHERPA_STT_ENCODER="$MODEL/encoder-epoch-99-avg-1.int8.onnx" \
SHERPA_STT_DECODER="$MODEL/decoder-epoch-99-avg-1.onnx" \
SHERPA_STT_JOINER="$MODEL/joiner-epoch-99-avg-1.int8.onnx" \
SHERPA_STT_TOKENS="$MODEL/tokens.txt" \
cargo test -p stt-sherpa \
  --test model_pipeline \
  vad_gated_wave_produces_a_final_transcript \
  -- --ignored --nocapture
```

## Default CI coverage

CI does not need model files or network access. It covers:

- Configuration validation and fixed Sherpa settings.
- Partial and final transcript behavior.
- Duplicate-hypothesis suppression.
- Protocol errors.
- Actor-thread ownership.
- Cancellation during native decoding.
- Stale queued-command rejection.
- Worker failure reporting.
- Compilation of both ignored model-backed tests.

To run the default crate suite locally:

```console
cargo test -p pipecrab-stt-sherpa
cargo test -p stt-sherpa
```
