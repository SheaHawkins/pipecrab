# Sherpa VAD adapter

## Goal

Add `pipecrab-vad-sherpa`, a native adapter that implements
`pipecrab_vad::VoiceActivityDetector` with Sherpa ONNX's Silero voice activity
detector, plus a `cpal` microphone example that prints `SpeechStarted` and
`SpeechStopped` as the user speaks.

The adapter is a segmenter-class detector and implements
`VoiceActivityDetector` directly. It does not use `SpeechScorer` or
`Debounced`; Sherpa already owns its threshold, minimum-speech, and
minimum-silence policy.

## Scope

This change includes:

- A publishable `crates/adapters/pipecrab-vad-sherpa` crate.
- A dedicated actor thread that exclusively owns one
  `sherpa_onnx::VoiceActivityDetector`.
- Arbitrary-length input support with exact 512-sample Silero inference
  windows and a retained partial-window remainder.
- Transition extraction from Sherpa's `detected()` state.
- Deterministic backend tests that do not require a model file.
- An ignored model-backed integration test.
- A desktop `examples/vad-sherpa` microphone example.
- User-facing setup and run instructions.

This change does not include:

- Sherpa speech-to-text.
- Model downloading inside the library or example.
- A change to `pipecrab-vad`'s detector lifecycle.
- WebAssembly support for the Sherpa adapter.
- Sharing one mutable Sherpa object between VAD and STT.

## Repository placement

The architecture assigns external engine adapters to `crates/adapters` and
already names `pipecrab-vad-sherpa` as the Sherpa implementation of the VAD
trait. The new manifest declares:

```toml
[package.metadata.pipecrab]
layer = "adapter"
```

The expected files are:

```text
crates/adapters/pipecrab-vad-sherpa/
├── Cargo.toml
├── CHANGELOG.md
├── src/
│   ├── backend.rs
│   ├── config.rs
│   ├── lib.rs
│   └── worker.rs
└── tests/
    └── detector.rs

examples/vad-sherpa/
├── Cargo.toml
└── src/main.rs
```

Add `pipecrab-vad-sherpa` and one exact Sherpa version to
`[workspace.dependencies]`. Both this adapter and the future STT adapter must
use the workspace Sherpa dependency without selecting their own version or
linkage features:

```toml
sherpa-onnx         = "=1.13.4"
pipecrab-vad-sherpa = { path = "crates/adapters/pipecrab-vad-sherpa", version = "0.2.0" }
```

Sherpa links statically by default. A build can provide its own matching native
library through `SHERPA_ONNX_LIB_DIR`; the adapter does not add another native
loading mechanism.

## Public API

The crate exports:

```rust
pub struct SherpaVad;
pub struct SherpaVadConfig;
pub enum SherpaVadBuildError;
pub trait Backend;
```

`SherpaVad` is a non-generic handle to the worker. It implements
`VoiceActivityDetector` and declares `AudioFormat::new(16_000, 1)`.

`SherpaVadConfig::new(model_path)` requires the model path and supplies these
production defaults:

| Setting | Default |
| --- | ---: |
| sample rate | 16,000 Hz |
| channels | 1 |
| Silero window | 512 samples |
| provider | `cpu` |
| Sherpa compute threads | 1 |
| threshold | 0.5 |
| minimum silence | 0.25 seconds |
| minimum speech | 0.25 seconds |
| maximum speech | 5 seconds |
| Sherpa result buffer | 30 seconds |
| debug logging | false |

The sample rate, channel count, window size, provider, and thread count are
fixed for this adapter. The speech policy and result-buffer duration remain
configurable. Construction validates the model path and finite, meaningful
numeric values before starting the worker.

`SherpaVadBuildError` distinguishes invalid configuration, model setup failure,
and worker spawn/setup failure. Runtime worker loss is reported through
`VadError::Engine` because `VoiceActivityDetector::process` fixes that error
type.

### Backend boundary

`Backend` is an object-safe, `Send + 'static` boundary with the operations the
worker needs:

```rust
pub trait Backend: Send + 'static {
    fn detected(&mut self) -> bool;
    fn accept_waveform(&mut self, samples: &[f32]);
    fn is_empty(&mut self) -> bool;
    fn pop(&mut self);
    fn reset(&mut self);
}
```

The production implementation contains exactly one
`sherpa_onnx::VoiceActivityDetector`. Test backends use the same boundary to
script transitions, record window lengths, and verify thread ownership without
loading Sherpa or a model.

The mutable receiver expresses the ownership rule even though Sherpa's Rust
wrapper takes `&self`. No backend reference escapes the actor.

## Worker ownership and protocol

`SherpaVad` contains:

- A command sender.
- An `Arc<AtomicU64>` reset generation.
- A worker handle that closes the command channel and joins the thread on drop.

The worker thread constructs, exclusively owns, accesses, and drops the
production backend. This guarantees serialized single-object use despite the
Sherpa wrapper implementing `Send + Sync`.

The data command is:

```rust
Process {
    samples: Arc<[f32]>,
    generation: u64,
    reply: futures::channel::oneshot::Sender<Result<Vec<VadEvent>, VadError>>,
}
```

`process()` snapshots the current generation, sends the shared sample buffer,
and awaits the reply. Sending does not copy the audio. A closed command channel
or reply channel becomes `VadError::Engine` rather than a panic.

The command channel may be unbounded because the PipeCrab stage awaits each
call and therefore has at most one normal VAD request in flight. `reset()` does
not enqueue a command.

### Reset and interruption

`VoiceActivityDetector::reset` must be synchronous, non-blocking, idempotent,
and infallible. `SherpaVad::reset()` satisfies that contract by incrementing the
atomic generation.

The worker compares the command generation with the current atomic generation:

- Before processing a command.
- Before every 512-sample window.
- Before publishing the reply.

When the generation changes, the worker calls `backend.reset()`, clears its
partial-window remainder, updates its observed generation, and abandons stale
work. This covers both important races:

1. An interrupt arrives after a process command is queued but before the worker
   starts it.
2. An interrupt arrives while the worker is handling a large input command.

The dropped `process()` future can therefore leave no speech state or remainder
in the next generation.

## Processing algorithm

The worker owns a `Vec<f32>` with capacity 512 for the partial window. It does
not append the entire incoming chunk to that vector.

For every process command:

1. If a remainder exists, copy only enough incoming samples to complete it.
2. Process the completed 512-sample window and clear the remainder.
3. Iterate over all remaining exact 512-sample slices without copying them.
4. Copy the final slice shorter than 512 into the remainder.
5. Return every transition in observation order.

Each exact window is handled as follows:

```rust
fn process_window(backend: &mut dyn Backend, window: &[f32], events: &mut Vec<VadEvent>) {
    let before = backend.detected();

    backend.accept_waveform(window);

    let after = backend.detected();
    match (before, after) {
        (false, true) => events.push(VadEvent::SpeechStarted),
        (true, false) => events.push(VadEvent::SpeechStopped),
        _ => {}
    }

    while !backend.is_empty() {
        backend.pop();
    }
}
```

Subdividing before feeding Sherpa preserves multiple transitions inside one
PipeCrab call. It also ensures the adapter obeys the trait's arbitrary-length
input contract while always presenting Silero with its expected window size.

Sherpa's completed segment samples are discarded because `VadStage` owns
pre-roll and forwards the original `AudioChunk`s. The adapter must not call
`front()` or copy a segment before popping it.

### Flush policy

Sherpa's `flush()` finalizes buffered trailing speech. It is not called during
normal processing because doing so after each PipeCrab chunk would terminate a
live utterance.

The current `VoiceActivityDetector` trait has no end-of-stream operation, and
`VadStage` cannot emit a data-lane transition from a synchronous `Stop` control
decision. This adapter therefore maps `reset()` but does not expose hidden
flush behavior. Finite-stream flushing requires an explicit detector lifecycle
design and is outside this change.

## Microphone example

The example pipeline is:

```text
CpalSource
    │ device-native sample rate, mono
    ▼
ResamplerStage: 16 kHz mono
    ▼
VadStage<SherpaVad>
    ▼
output drain: print edges and discard gated audio
```

`CpalSource` captures at the input device's default sample rate. The example
places one `ResamplerStage` before VAD because `VadStage` enforces the
detector's 16 kHz mono format before any sample reaches Sherpa.

The input pump follows the existing echo example: send `SystemFrame::Start`,
then send each captured `AudioChunk` until capture ends or an optional
`--seconds` limit expires. The output pump handles:

```rust
Received::Data(DataFrame::SpeechStarted) => println!("SpeechStarted"),
Received::Data(DataFrame::SpeechStopped) => println!("SpeechStopped"),
Received::Data(DataFrame::Audio(_)) => {},
Received::Data(_) | Received::Sys(_, _) => {},
```

Consuming the gated audio is required even though the example does not use it;
otherwise downstream backpressure would stop the pipeline.

The executable accepts:

```console
cargo run -p vad-sherpa -- --model ./silero_vad.onnx
cargo run -p vad-sherpa -- --model ./silero_vad.onnx --seconds 10
```

README instructions include the official model location:

```console
curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
  -o silero_vad.onnx
```

Expected interactive output includes the selected microphone, its capture
rate, the 16 kHz processing format, and alternating edge names.

## Tests

### Backend-free tests

Use a scripted `Backend` to cover:

- Fewer than 512 samples are retained and do not reach the backend.
- Split input such as 300 then 212 produces one exact window.
- Large input produces only 512-sample backend calls.
- A trailing remainder survives into the next command.
- A single command can return several alternating transitions.
- A completed-segment queue is fully popped after every window.
- `reset()` clears both backend state and the adapter remainder.
- A reset before a queued command prevents that command from being processed.
- A reset during a large command stops stale work at a window boundary.
- A stale reply is not published as a current-generation result.
- Every backend operation occurs on the same worker thread.
- Worker setup failure returns `SherpaVadBuildError`.
- Worker termination maps to `VadError::Engine`.
- Dropping `SherpaVad` closes and joins its worker.
- `input_format()` is exactly 16 kHz mono.

### Model-backed test

Add an ignored integration test that reads `SHERPA_VAD_MODEL`, constructs the
production backend, processes representative audio, and verifies that the
worker remains healthy. It is ignored because the workspace does not vendor a
model or microphone fixture.

### Example verification

Run the microphone example manually and verify:

- Silence produces no printed edges.
- Speaking prints `SpeechStarted`.
- Remaining silent beyond the configured duration prints `SpeechStopped`.
- Repeated utterances produce alternating start/stop pairs.
- Capture overruns remain zero during normal use.

## Sherpa STT follow-on

Cargo dependency unification gives VAD and STT one Sherpa Rust package and one
linked native library in the application. It does not share models, ONNX
sessions, mutable objects, actor threads, or CPU budgets.

For one session, the target topology is:

```text
cpal or transport
    ▼
one 16 kHz mono resampler
    ▼
VAD actor
  one VoiceActivityDetector
  one Sherpa compute thread
    ▼
VadStage gate
    ▼
SttStage
    ▼
STT actor
  one OnlineRecognizer
  one active OnlineStream
  one Sherpa compute thread initially
    ▼
transcripts
```

The actors remain separate. STT decoding is heavier than a Silero window and
must not hold the VAD actor's queue. The operating system can preempt separate
workers, while the explicit one-thread Sherpa configurations avoid multiplying
internal compute pools. Increase the STT thread count only from real-time-factor
and queue-delay benchmarks on target hardware.

The audio conversion and buffers are shared without coupling the native model
objects:

- Audio is resampled once before VAD.
- `VadStage` forwards the original `Arc`-backed chunks, including its pre-roll.
- STT receives those shared chunks without another sample-buffer copy.
- Both adapters declare and validate the same 16 kHz mono format.

The future STT worker owns the `OnlineRecognizer` and its current
`OnlineStream`. `SpeechStarted` creates or resets the stream, audio feeds it,
and `SpeechStopped` calls `input_finished()` and drains final decode steps.
Interrupt generation invalidates the current stream in the same way VAD reset
invalidates stale process work.

### Multiple sessions

Do not allocate one complete STT recognizer and model per session. Sherpa's
online recognizer can create several streams and decode multiple streams in one
batch. `pipecrab-stt-sherpa` should therefore grow a shared pool API:

```text
per-session SttStage
    ▼
per-session StreamingTranscriber handle
    ▼
shared STT actor
  one OnlineRecognizer
  OnlineStream per session ID
  ready-stream batching
  bounded decode scheduling
```

Each session handle still represents one active utterance and satisfies the
`StreamingTranscriber` protocol. The shared actor owns the recognizer, model
weights, stream map, and batching policy. VAD remains per session because every
detector has independent speech state and a partial-window remainder.

CI should reject duplicate Sherpa packages with a metadata test or equivalent
`cargo tree -d` assertion. This ensures both adapters use the same pinned
Sherpa build while leaving resource scheduling explicit.

## Documentation changes

Update the root README with:

- `pipecrab-vad-sherpa` in the adapter list.
- The model download command.
- The microphone example command.
- The required 16 kHz resampling topology.
- A note that the first Sherpa build may obtain matching native libraries.

Crate-level documentation explains actor ownership, the fixed input format,
window retention, queue draining, reset generations, and why Sherpa segment
audio is discarded.

## Verification

Run:

```console
cargo fmt --all -- --check
cargo test -p pipecrab-vad-sherpa
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo tree -d
cargo run -p vad-sherpa -- --model ./silero_vad.onnx
```

The architecture layering test must recognize both the adapter and example
through their declared `adapter` and `app` layers without any dependency
exception.

## Acceptance criteria

- The worker exclusively owns one production Sherpa VAD object.
- Sherpa sees only 512-sample windows.
- A partial window is retained across process calls and cleared on reset.
- Every `detected()` transition is returned in order, including multiple edges
  from one PipeCrab call.
- Sherpa's completed-segment queue never accumulates retained audio.
- `reset()` is synchronous, non-blocking, idempotent, and prevents stale worker
  state from crossing an interrupt.
- The detector declares 16 kHz mono and `VadStage` rejects any other format.
- The example resamples device capture once and prints both edge types from a
  real microphone.
- Default workspace tests do not require a model file or audio hardware.
- The workspace contains one resolved Sherpa version and linkage selection.
- The design permits a separate STT actor and a future shared multi-stream STT
  pool without changing the VAD adapter's public contract.
