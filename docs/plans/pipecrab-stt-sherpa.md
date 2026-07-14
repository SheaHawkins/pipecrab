# Sherpa streaming STT adapter

## Goal

Add `pipecrab-stt-sherpa`, a native adapter that implements
`pipecrab_stt::StreamingTranscriber` directly with Sherpa ONNX's
`OnlineRecognizer` and `OnlineStream`.

The adapter performs incremental decoding while audio arrives. It does not use
the one-shot `Transcriber` trait or the `Buffered` adapter.

## Scope

This change includes:

- A publishable `crates/adapters/pipecrab-stt-sherpa` crate.
- A dedicated actor thread that exclusively owns one `OnlineRecognizer` and at
  most one active `OnlineStream`.
- Direct implementations of `begin_utterance`, `feed`, `end_utterance`, and
  `cancel`.
- Generation-tagged commands and cancellation checks between native decode
  steps.
- Partial transcripts whose stable prefix is always zero.
- Deterministic backend tests that do not require model files.
- An ignored model-backed integration test.

This change does not include:

- A one-shot Sherpa `Transcriber` implementation.
- Sherpa endpoint detection; PipeCrab's VAD edges remain authoritative.
- Model downloading or model assets in the crate.
- Hotwords, an external language model, or beam search.
- Pipeline-capacity tuning.
- Core pinning or custom OS scheduler policy.
- A shared recognizer pool for multiple sessions.
- WebAssembly support for the Sherpa adapter.

## Repository placement

The adapter belongs under `crates/adapters` and declares the adapter layer:

```toml
[package.metadata.pipecrab]
layer = "adapter"
```

The expected files are:

```text
crates/adapters/pipecrab-stt-sherpa/
├── Cargo.toml
├── src/
│   ├── backend.rs
│   ├── config.rs
│   ├── lib.rs
│   └── worker.rs
└── tests/
    └── transcriber.rs
```

Add the adapter to `[workspace.dependencies]` and use the workspace's exact
Sherpa version:

```toml
pipecrab-stt-sherpa = { path = "crates/adapters/pipecrab-stt-sherpa", version = "0.3.0" }
sherpa-onnx         = "=1.13.4"
```

Both Sherpa adapters use this one dependency declaration so applications
resolve one Sherpa Rust package and native library version.

## Public API

The crate exports:

```rust
pub struct SherpaStt;
pub struct SherpaSttConfig;
pub enum SherpaSttBuildError;
pub trait Backend;
```

`SherpaStt` is a non-generic worker handle. It implements
`StreamingTranscriber` and declares `AudioFormat::new(16_000, 1)`.

Construction APIs are:

```rust
impl SherpaStt {
    pub fn new(config: SherpaSttConfig) -> Result<Self, SherpaSttBuildError>;
    pub fn with_backend(backend: impl Backend) -> Result<Self, SherpaSttBuildError>;
}
```

`with_backend` supports deterministic tests while preserving the same actor
ownership boundary as the production recognizer.

`SherpaSttBuildError` distinguishes invalid configuration, recognizer creation
failure, and worker spawn or setup failure. Runtime worker loss maps to
`SttError::Engine`.

## Configuration

`SherpaSttConfig::new` requires paths to the streaming transducer files:

- Encoder.
- Decoder.
- Joiner.
- Tokens.

The production defaults are:

| Setting | Default |
| --- | ---: |
| sample rate | 16,000 Hz |
| channels | 1 |
| provider | `cpu` |
| decoding method | `greedy_search` |
| endpoint detection | disabled |
| Sherpa compute threads | 2 |
| debug logging | false |
| hotwords | disabled |
| external language model | disabled |

The sample rate, channel count, provider, decoding method, and endpoint policy
are fixed for this adapter. `num_threads` and debug logging remain configurable.
The thread count must be positive. A balanced mobile profile uses two threads;
a thermal or low-power profile can select one. Three threads are an explicit
benchmark option rather than a production assumption.

Construction validates that all required paths identify files and contain
valid UTF-8 before starting the worker. It then produces an
`OnlineRecognizerConfig` equivalent to:

```rust
config.model_config.num_threads = 2;
config.model_config.provider = Some("cpu".into());
config.decoding_method = Some("greedy_search".into());
config.enable_endpoint = false;
```

The first English mobile model target is
`sherpa-onnx-streaming-zipformer-en-20M-2023-02-17`, fixed to batch size one,
with its int8 encoder. The adapter accepts explicit paths and does not infer or
download a particular model package.

## Backend boundary

`Backend` exposes only the serialized recognizer operations required by the
actor. Its associated stream type lets tests substitute an ordinary Rust value
for `OnlineStream`:

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

The production backend contains exactly one `OnlineRecognizer`. Its stream is
`sherpa_onnx::OnlineStream`, and `accept_waveform` always supplies the fixed
16,000 Hz sample rate.

Mutable receivers express exclusive actor ownership even where Sherpa's Rust
wrapper exposes shared-reference methods. Neither the recognizer nor a stream
reference escapes the actor.

## Worker ownership and commands

`SherpaStt` contains:

- A command sender.
- An `Arc<AtomicU64>` cancellation generation.
- A worker handle that closes the command channel and joins the thread on drop.

The worker state is equivalent to:

```rust
struct SttWorker<B: Backend> {
    recognizer: B,
    stream: Option<B::Stream>,
    generation: u64,
    last_partial: String,
}
```

For the production backend, `recognizer` owns the Sherpa `OnlineRecognizer`.
The worker constructs, accesses, and drops the recognizer and stream on its one
actor thread.

Commands are:

```rust
Begin {
    generation: u64,
    reply: oneshot::Sender<(u64, Result<(), SttError>)>,
}

Feed {
    samples: Arc<[f32]>,
    generation: u64,
    reply: oneshot::Sender<(u64, Result<Vec<SttEvent>, SttError>)>,
}

End {
    generation: u64,
    reply: oneshot::Sender<(u64, Result<Vec<SttEvent>, SttError>)>,
}
```

Each public async method snapshots the current generation, submits its command,
and awaits the reply. Sending a feed command moves the shared `Arc<[f32]>`
without copying its sample buffer. Closed command and response channels become
`SttError::Engine`.

## Utterance protocol

### Begin

`begin_utterance()` creates a fresh stream, clears `last_partial`, and decodes
the zero-valued startup context configured on `SherpaSttConfig`. The default is
one second. Calling it while a stream is active is a protocol error. It never
resets or silently replaces an active utterance.

### Feed

Feeding without an active stream is a protocol error. An accepted feed follows
this sequence:

```rust
stream.accept_waveform(16_000, &samples);

while recognizer.is_ready(stream) {
    if generation_is_stale() {
        drop_active_stream();
        return;
    }

    recognizer.decode(stream);

    if generation_is_stale() {
        drop_active_stream();
        return;
    }
}

let text = recognizer
    .get_result(stream)
    .map(|result| result.text)
    .unwrap_or_default();

if text != last_partial {
    last_partial.clone_from(&text);
    emit_partial(text, 0);
}
```

One `decode` call is one cancellation boundary. The worker checks the generation
before and after every step. It does not batch ready streams or ask Sherpa to
drain the stream in one native call.

Every changed hypothesis is emitted, including a retraction to an empty
hypothesis. Repeated identical hypotheses produce no event. All partials use
`stable: 0` because Sherpa does not guarantee that any current prefix is
permanent.

### End

Ending without an active stream is a protocol error. `end_utterance()` appends
the configured zero-valued final context, which defaults to 300 milliseconds,
marks the stream complete, drains ready decode steps with the same generation
checks, and returns exactly one final event:

```rust
stream.accept_waveform(16_000, &final_padding);
stream.input_finished();

while recognizer.is_ready(stream) {
    recognizer.decode(stream);
}

let final_text = recognizer
    .get_result(stream)
    .map(|result| result.text)
    .unwrap_or_default();

drop(stream);
return SttEvent::Final(final_text.into());
```

The active stream and `last_partial` are cleared before another utterance can
begin. Sherpa endpoint state is never consulted because `SpeechStopped` is the
authoritative utterance boundary.

## Cancellation

`StreamingTranscriber::cancel` is synchronous, non-blocking, idempotent, and
infallible. It increments the atomic generation and does not enqueue a command
or wait for the actor.

The worker compares each command's generation with the atomic generation:

- Before touching a queued command.
- Before every decode step.
- After every decode step.
- Before publishing a hypothesis or final result.

When the generation changes, the worker drops the active stream, clears
`last_partial`, and records the new generation. Commands tagged with an older
generation are skipped without touching recognizer state. The handle also
compares the response generation after awaiting so a cancellation racing with
reply delivery cannot publish stale output.

This covers these races:

1. Cancellation arrives after a command is queued but before the actor reads
   it.
2. Cancellation arrives between ready decode steps.
3. Cancellation arrives while one native `decode()` call is running.
4. Cancellation arrives after the actor sends a reply but before the caller
   observes it.

Sherpa cannot interrupt a `decode()` already executing in native inference.
That call completes, after which the generation check drops the stream and
discards its result. Responsiveness therefore depends on small upstream audio
chunks, one decode step per check, and a small Sherpa inference pool.

## Resource schedule

The intended single-session schedule is:

| Execution context | Count | Work | Configuration |
| --- | ---: | --- | --- |
| Audio callback | Platform-owned | Copy PCM into a lock-free ring | Real-time; no allocation, inference, channels, or locks |
| PipeCrab driver | 1 | Pipeline stages and pumps | Must not block |
| VAD actor | 1 | 512-sample Silero steps | Sherpa threads = 1 |
| STT actor | 1 | Own recognizer and active stream | Normal foreground scheduling |
| STT inference pool | 1–2 initially | ONNX Runtime kernels | 2 balanced, 1 low-power |

Workers are not pinned to cores. Thread priority and thermal policy remain with
the operating system. Target-device benchmarks determine whether a three-thread
STT experiment improves sustained latency.

## Tests

### Backend-driven tests

A scripted backend covers:

- `input_format()` is exactly 16 kHz mono.
- Begin creates one stream and a double begin is rejected.
- Feed and end without an active utterance are rejected.
- Feed passes each shared chunk to the active stream.
- Each ready state causes one decode step.
- A changed hypothesis emits `Partial { stable: 0 }`.
- An unchanged hypothesis emits no duplicate partial.
- A hypothesis can retract to empty.
- End appends final context, marks input finished, drains every ready step, and
  emits one final.
- Empty Sherpa results become an empty partial or final according to protocol
  state.
- A completed utterance drops its stream and permits a clean next begin.
- Every backend operation runs on one actor thread.
- Cancellation before a queued command prevents backend access.
- Cancellation between decode steps stops further decoding.
- Cancellation during a blocked decode discards that decode's hypothesis.
- Commands from an old generation do not affect the next utterance.
- A response racing with cancellation is suppressed.
- Worker termination maps to `SttError::Engine`.
- Dropping `SherpaStt` closes and joins the actor.

The tests use small chunks and controllable blocking points so cancellation
races are deterministic rather than timing-dependent.

### Configuration tests

Configuration tests cover:

- Missing encoder, decoder, joiner, and tokens files.
- Non-UTF-8 paths where the platform permits them.
- Zero and negative thread counts.
- CPU provider, greedy decoding, and disabled endpointing in the translated
  Sherpa configuration.
- Two compute threads as the default.

### Model-backed test

Add an ignored integration test that reads explicit model paths from
environment variables, constructs `SherpaStt`, feeds representative 16 kHz
audio, and completes an utterance. Default workspace tests do not require model
files or network access.

## Verification

Run:

```console
cargo fmt --all -- --check
cargo test -p pipecrab-stt-sherpa
cargo test -p pipecrab-arch --test layering
cargo clippy -p pipecrab-stt-sherpa --all-targets -- -D warnings
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo tree -d
```

The architecture test must accept the new adapter without a dependency-layer
exception. `cargo tree -d` should continue to show one resolved Sherpa version.

## Acceptance criteria

- `SherpaStt` implements `StreamingTranscriber` directly.
- The production actor exclusively owns one `OnlineRecognizer` and at most one
  active `OnlineStream`.
- Begin creates and primes a stream, feed incrementally decodes it, and end
  appends final context, flushes, and drops it.
- Sherpa endpoint detection is disabled and PipeCrab VAD edges define every
  utterance boundary.
- Changed hypotheses are emitted as partials with `stable: 0`; identical
  hypotheses are suppressed.
- End emits exactly one final result, including an empty result when Sherpa has
  no text.
- Cancel is synchronous and does not wait for native inference.
- Stale queued commands, in-progress decode results, and racing replies cannot
  cross a cancellation generation.
- Protocol violations and worker loss surface as `SttError::Engine`.
- The adapter defaults to CPU greedy search with two compute threads and allows
  one-thread low-power configuration.
- Tests exercise streaming and cancellation deterministically without a model.
- Default workspace validation requires neither model files nor audio hardware.
