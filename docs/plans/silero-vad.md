# Plan: the `silero-vad` crate family

Status: **draft for review** · 2026-07-08

This document locks the interfaces, dependency strategy, and rationale for
pipecrab's Silero voice-activity-detection crates before implementation, so the
design doesn't drift. Every locked decision cites what it references — a file
in this repo, or an external source with an exact version.

## 1. Decisions at a glance

| Decision | Choice |
|---|---|
| Architecture | Backend-agnostic sans-IO core + thin per-target backends |
| Crate names | `silero-vad` (core), `silero-vad-ort` (native), `silero-vad-web` (browser); rename the published `silero-ort`/`silero-web` scaffolds |
| Model version | Silero VAD v6.2.x, v5+ ONNX contract |
| Native inference | `ort` 2.0.0-rc.12 (caret, never `=`); app owns linking |
| Web inference | transformers.js (`@huggingface/transformers` 4.2.0) via an inference-only JS shim — one substrate and one version shared with `moonshine-web`/`kokoro-web`; the pipeline is Rust end-to-end up to that boundary |
| Model delivery (native) | Bundled via `include_bytes!` behind a default feature + BYO escape hatches |
| Model delivery (web) | Fetched from HF Hub by default (`onnx-community/silero-vad`), self-hostable |
| Frame delivery | `VadStage` aggregates to exact 512-sample frames in `decide`; engines are strict about frame length |
| Correctness bar | Frame-level numerical parity with the reference Python package, fixture-tested |

## 2. What problems these crates solve (vs. each prior art)

The honest rationale is **not** "existing crates pin stale `ort`" — one of them
doesn't. It is: **no existing crate separates the Silero model logic (where all
the correctness risk lives) from the inference backend, and our browser target
makes that separation mandatory.** The model logic — context prefixing, state
lifecycle, hysteresis segmentation, offline splitting — is several hundred
subtle lines. The inference call is ~20 lines per backend. Prior art welds the
20 lines to the several hundred.

### vs. `voice_activity_detector` 0.2.1 (nkeenan38)

- **Exact-pins `ort = "=2.0.0-rc.10"` and `ort-sys = "=2.0.0-rc.10"`.**
  Because `ort-sys` declares `links = "onnxruntime"`, two different exact pins
  in one dependency graph are a *hard build error*, not just friction. We use a
  caret bound that unifies across rc.11/rc.12/2.0.0-final.
- **Skips the v5+ context prefix**: feeds bare `[1, 512]` windows, relying only
  on the state tensor, so its probabilities silently diverge from the Python
  reference. Our core owns the prefix and is fixture-tested against reference
  output.
- **Conflates hangover, padding, and minimum duration into one
  `padding_chunks` knob** — no `neg_threshold` hysteresis, no
  `min_speech_duration`, so single-chunk blips register as speech. We ship
  Python-parity segmentation.
- No browser story.

### vs. `silero-vad-rust` 6.2.1 (sheldonix)

Current model (tracks Silero v6.2.1) and a caret `ort` bound — the "stale ort"
argument does **not** apply to it. The problems are architectural:

- **Welded to `ort`**: its `VadIterator`/`get_speech_timestamps` are coupled to
  its ort-backed model struct, so nothing is reusable for a browser build. Under
  a "use it" plan we would still write all the hard parts for web, then maintain
  two numerical codepaths that can drift.
- **Defaults to `ort/load-dynamic`.** Cargo features are additive across the
  graph and `load-dynamic` disables all compile-time linking, so any app pulling
  it with default features is forced into "user must supply libonnxruntime
  1.22.x" mode — even if the app wanted static `download-binaries`. A library
  should never impose a linking mode; ours doesn't.
- Single-maintainer model-update cadence in our hot path.
- Its port is MIT: we may extract from its tested `get_speech_timestamps`
  implementation with attribution as an accelerant.

### vs. `silero-vad-rs` 0.1.2 (binarycrayon)

- Exact-pins `ort = "=2.0.0-rc.9"` (guaranteed conflict with rc.10+ consumers),
  v5-era, user-supplied model path only (worst DX of the three), lightly
  maintained. Useful only as a compact port reference.

### vs. `@ricky0123/vad-web` 0.0.30 (JavaScript)

The established browser Silero VAD — but JS-only (no Rust API), and it binds
`onnxruntime-web` directly, which in our stack would ship a **second**
differently-versioned ort-wasm runtime alongside the one transformers.js
already loads for Moonshine/Kokoro. It validates the browser pattern; it can't
be our engine.

**What none of them provide:** a sans-IO core giving native and web the *same*
numerics from *one* implementation, an API that never dictates the app's `ort`
version or linking mode, and Python-parameter-parity segmentation available
without an inference dependency.

(Side note settled: the `-rust`/`-rs` suffixes in prior art were not forced —
`silero-vad` is unclaimed on crates.io as of 2026-07-08, verified via the
sparse index. The suffixes mirror GitHub repo naming; we don't repeat that.)

## 3. Crate family and fit with ARCHITECTURE.md

```
silero-vad            core: ALL model logic; no inference deps, no async, wasm-clean
   ▲            ▲
silero-vad-ort  silero-vad-web     engines: pipecrab-free, cfg-selected per target
   ▲            ▲                  (ort natively · transformers.js in the browser)
pipecrab-vad-silero                model crate: adapts an engine to pipecrab-vad
```

**Does transformers.js invalidate the crating strategy? No — it fulfills it.**
`ARCHITECTURE.md` already prescribes cfg-selected, pipecrab-free engines and
names transformers.js as the browser substrate for STT ("`moonshine-web` via
transformers.js", ARCHITECTURE.md:56; "browser Transformers.js in a Worker",
ARCHITECTURE.md:78). Routing Silero through transformers.js as well makes the
web story *universal*: one `@huggingface/transformers` runtime (and one
onnxruntime-web wasm download underneath it) serves `silero-vad-web`,
`moonshine-web`, and `kokoro-web`. This is proven upstream — Hugging Face's
official `conversational-webgpu` example runs Silero VAD + STT + Kokoro in a
single worker on transformers.js, and their `moonshine-web` example is
literally Silero VAD + Moonshine.

Two documents currently disagree on the moonshine-web substrate
(`crates/moonshine-web/src/lib.rs` says "onnxruntime-web"; ARCHITECTURE.md:56
says transformers.js). This plan resolves the conflict in favor of
**transformers.js everywhere on web**; the moonshine-web doc comment should be
updated when that crate is implemented.

What this plan *adds* to the architecture rather than changes:

- A new tier: the **engine-core crate** (`silero-vad`) — pipecrab-free *and*
  backend-free. Dependencies still point strictly downward
  (engine → engine-core; ARCHITECTURE.md:12). The naming rule is preserved:
  no `pipecrab-` prefix ⇒ useful standalone (ARCHITECTURE.md:58-61).
- **Renames**: `silero-ort` → `silero-vad-ort`, `silero-web` →
  `silero-vad-web`. "Silero" is a company with STT/TTS model families; the
  model's own name is "Silero VAD", and it's how users search. The published
  0.1.0 scaffolds are yanked after the new names publish (names stay claimed,
  each with a final README pointer); workspace members, ARCHITECTURE.md:22/38,
  the CI wasm gate list, and the release-plz graph update accordingly.
- The `kokoro-*`/`moonshine-*` families can adopt the same engine-core split
  later if their logic warrants it; nothing here requires it.

## 4. Locked interfaces

### 4.1 `silero-vad` — the sans-IO core

References: model I/O contract from `snakers4/silero-vad` (v6.2.1) —
`src/silero_vad/utils_vad.py` (OnnxWrapper) and the repo's C++ ONNX example;
tensor shapes verified against both. Parameter vocabulary and defaults from
`utils_vad.py` (`VADIterator`, `get_speech_timestamps`).

Model contract being encoded (v5/v6 ONNX):

| Tensor | Name | Shape / dtype | Notes |
|---|---|---|---|
| in | `input` | `[1, N+ctx]` f32 | N = 512 @ 16 kHz / 256 @ 8 kHz; ctx = 64 / 32 samples of the previous frame (zeros first) |
| in | `state` | `[2, 1, 128]` f32 | zeros on reset |
| in | `sr` | scalar i64 | 16000 or 8000 |
| out | `output` | `[1, 1]` f32 | speech probability |
| out | `stateN` | `[2, 1, 128]` f32 | feed back as `state` |

The core is sync-agnostic **by construction** (sans-IO), because the native
backend is synchronous (`ort::Session::run` takes `&mut self` in rc.12) while
the web backend is inherently async (JS promises). The core never calls a
backend; backends drive it:

```rust
pub enum SampleRate { Hz8000, Hz16000 }
// frame_len(): 256 | 512 · context_len(): 32 | 64
pub const STATE_LEN: usize = 256; // [2, 1, 128] flattened

/// Owns recurrent state + context tail. Validates frames, stages exact model
/// inputs, receives the model's state write-back. No I/O, no inference.
pub struct SileroSession { /* state, context, input buffer, rate */ }

impl SileroSession {
    pub fn new(rate: SampleRate) -> Self;
    /// `frame.len()` must equal `frame_len()` — anything else is
    /// `FrameError::InvalidLength` (mirrors the reference's hard error).
    /// Returns the staged, context-prefixed tensors for one inference step.
    pub fn stage(&mut self, frame: &[f32]) -> Result<Staged<'_>, FrameError>;
    pub fn reset(&mut self);
}

/// Backend contract: run the graph with these named tensors
/// (`input`, `state`, `sr`), write `stateN` back into `state`, return `output`.
pub struct Staged<'s> {
    pub input: &'s [f32],               // len = frame_len + context_len
    pub state: &'s mut [f32; STATE_LEN],
    pub sr: i64,
}
```

A sync convenience engine for sync backends (native), so standalone users get
an ergonomic API without pipecrab and without async:

```rust
pub trait SileroModel {
    type Error;
    fn infer(&mut self, input: &[f32], state: &mut [f32; STATE_LEN], sr: i64)
        -> Result<f32, Self::Error>;
}

pub struct SileroVad<M: SileroModel> { /* SileroSession + M */ }
impl<M: SileroModel> SileroVad<M> {
    pub fn new(model: M, rate: SampleRate) -> Self;
    pub fn process(&mut self, frame: &[f32]) -> Result<f32, Error<M::Error>>;
    pub fn reset(&mut self);
}
```

Async backends (web) use `SileroSession::stage` directly across an `.await`;
they get the identical numerics because the staging/state code is shared.

Pure segmentation, usable with *any* probability source (no inference dep):

```rust
/// Streaming — parity with Python VADIterator (defaults in parentheses).
pub struct StreamingOptions {
    pub threshold: f32,                 // 0.5
    pub neg_threshold: Option<f32>,     // None ⇒ max(threshold − 0.15, 0.01)
    pub min_silence_duration_ms: u32,   // 100
    pub speech_pad_ms: u32,             // 30
}

pub enum VadEvent { SpeechStart { sample: u64 }, SpeechEnd { sample: u64 } }

pub struct Segmenter { /* rate, opts, triggered/temp_end/current_sample */ }
impl Segmenter {
    pub fn new(rate: SampleRate, opts: StreamingOptions) -> Self;
    /// One probability per processed frame; at most one event per push,
    /// matching the reference. `reset()` clears state machine AND counters.
    pub fn push(&mut self, probability: f32) -> Option<VadEvent>;
    pub fn flush(&mut self) -> Option<VadEvent>;
    pub fn reset(&mut self);
}

/// Offline — parity with Python get_speech_timestamps; adds
/// min_speech_duration_ms (250), max_speech_duration_s (∞) + force-split.
pub struct SpeechSegment { pub start: u64, pub end: u64 } // samples
pub fn get_speech_timestamps<M: SileroModel>(
    audio: &[f32], vad: &mut SileroVad<M>, opts: &TimestampOptions,
) -> Result<Vec<SpeechSegment>, Error<M::Error>>;
```

Plus a small `FrameBuffer` (push arbitrary-length sample slices, pop exact
frames) so *standalone* callers with hardware-sized chunks don't hand-roll
windowing. Inside pipecrab it is not used: `VadStage` owns framing (§4.5).

The Python parameter vocabulary is kept **verbatim** — that's the API
compatibility that matters (users' tuning knowledge transfers); symbol-level
compatibility with other Rust crates has no value.

Cargo: no dependencies (std only). Optional `bundled-model` feature embeds
`silero_vad.onnx` (op16, 2,327,524 bytes, MIT) so both backends share one copy
of the bytes; well under crates.io's 10 MB package limit.

### 4.2 `silero-vad-ort` — native engine

References: `ort` 2.0.0-rc.12 (`Session::run` takes `&mut self`; `Session` is
`Send + Sync`, not `Clone`; `TensorRef::from_array_view` gives zero-copy
borrowed inputs). MSRV note below.

```rust
pub struct OrtModel { session: ort::Session }

impl OrtModel {
    /// BYO — the app fully controls ort setup; we never dictate it.
    pub fn from_session(session: ort::Session) -> Self;
    pub fn from_bytes(model: &[u8]) -> Result<Self, ort::Error>;
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ort::Error>;
    #[cfg(feature = "bundled-model")]
    pub fn bundled() -> Result<Self, ort::Error>; // core's embedded op16 model
}

impl silero_vad::SileroModel for OrtModel { /* TensorRef in, extract out */ }

/// Convenience alias + constructors mirroring the core engine.
pub type SileroVadOrt = silero_vad::SileroVad<OrtModel>;

pub use ort;         // apps configure linking/EPs against OUR ort version
pub use silero_vad;  // one import for standalone users
```

Session defaults: CPU only (no execution-provider features — GPU transfer
overhead exceeds compute for a ~1 ms/frame model), `with_intra_threads(1)`.

### 4.3 `silero-vad-web` — browser engine

References: Hugging Face official examples
`transformers.js-examples/moonshine-web/src/worker.js` and
`conversational-webgpu/src/worker.js` (the `AutoModel` +
`config: { model_type: "custom" }` pattern with manual `state` feedback);
model repo `onnx-community/silero-vad` (fp32 2.24 MB, fp16 1.15 MB,
int8 0.64 MB); `@huggingface/transformers` 4.2.0.

A thin JS shim (`js/silero.js`) wraps transformers.js; Rust binds it with
`#[wasm_bindgen(module = "/js/silero.js")]` + `wasm_bindgen_futures::JsFuture`:

```rust
pub struct WebModel { /* JsValue handle to the loaded AutoModel */ }

impl WebModel {
    /// Default: fetch onnx-community/silero-vad from HF Hub (browser-cached).
    pub async fn from_hub() -> Result<Self, WebError>;
    /// Self-hosted/offline: transformers.js env.localModelPath root.
    pub async fn from_local(base_url: &str) -> Result<Self, WebError>;

    pub async fn infer(&self, input: &[f32], state: &mut [f32; STATE_LEN], sr: i64)
        -> Result<f32, WebError>;
}

/// Async engine mirroring silero_vad::SileroVad, built on SileroSession.
pub struct SileroVadWeb { /* SileroSession + WebModel */ }
impl SileroVadWeb {
    pub async fn process(&mut self, frame: &[f32]) -> Result<f32, WebError>;
    pub fn reset(&mut self);
}
```

Notes that are part of the locked design:

- **We feed the context-prefixed `[1, 576]` input on web too.** The official JS
  examples feed bare `[1, 512]` and therefore diverge numerically from the
  Python reference (same bug class as `voice_activity_detector`); the graph
  accepts both lengths, and the Python wrapper feeds 576 to this same graph.
  Our core makes web and native identical.
- Per-chunk cost is fine: 2 KB crossing Rust→JS every 32 ms, ~µs of interop
  against ~1 ms of inference — the official demos already cross
  worklet→worker per chunk.
- `silero-vad-web` is in the CI wasm gate (`.github/workflows/ci.yml`), like
  the `silero-web` name it replaces.

### 4.4 `pipecrab-vad-silero` — the model crate

References: `crates/pipecrab-vad/src/lib.rs` (`VoiceActivityDetector`,
`VadVerdict`, `VadError` — `detect(&self, …)`, `MaybeSendSync` bound, rejects
format mismatches with `UnsupportedFormat`, never resamples);
`crates/pipecrab-vad/src/stage.rs` (`VadStage` owns start/stop debouncing);
`crates/pipecrab-runtime/src/offload.rs` (`offload` is `Send + 'static`;
the wasm implementation is currently a stub).

```rust
pub struct SileroDetectorOptions {
    pub threshold: f32,        // 0.5 → VadVerdict::is_speech = prob ≥ threshold
    pub sample_rate: u32,      // 16_000; expected AudioFormat is (rate, mono)
}

pub struct SileroDetector { /* Mutex<engine>, opts */ }

// native
impl SileroDetector {
    pub fn new() -> Result<Self, VadError>;                       // bundled model
    pub fn with_options(opts: SileroDetectorOptions) -> Result<Self, VadError>;
    pub fn from_model(model: silero_vad_ort::OrtModel, opts: …) -> Self;
}
// wasm32
impl SileroDetector {
    pub async fn from_hub() -> Result<Self, VadError>;
    pub async fn from_local(base_url: &str, opts: …) -> Result<Self, VadError>;
}

// The trait impl uses the explicit cfg_attr async_trait pair, like the
// ScriptedVad precedent in crates/pipecrab-vad/tests/vad_stage.rs.
impl VoiceActivityDetector for SileroDetector {
    async fn detect(&self, samples: &[f32], format: AudioFormat)
        -> Result<VadVerdict, VadError>;
}
```

Locked `detect` semantics:

- Reject `format != AudioFormat::new(16_000, 1)` with
  `VadError::UnsupportedFormat` (no resampling — trait contract).
- **Strict frame length**: exactly `frame_len()` samples (512 @ 16 kHz) per
  call, one frame in → one verdict out. Buffering is the temporal analog of
  resampling, which this trait already refuses; inside pipecrab, `VadStage`
  supplies exact frames (§4.5), so a wrong length reaching the detector is a
  wiring bug, rejected with `VadError::InvalidFrameLen { expected, got }`
  (new variant, §4.5).
- Threshold only — no hysteresis here: `VadStage` already owns edge debouncing
  (`start_windows`/`stop_windows`); layering the core `Segmenter` on top would
  double-debounce.
- Native: inference runs inside `offload(move || …)` (owned samples +
  `Arc<Mutex<…>>` cross the `Send + 'static` boundary) so barge-in can preempt.
  wasm32: runs inline — `offload` on wasm is not implemented yet, and one frame
  is ~1 ms; revisit if a Web Worker offload lands in `pipecrab-runtime`.
- Engine mutability lives here: the standalone crates keep honest `&mut`
  (matching `Session::run`); this crate's `Mutex` satisfies the trait's `&self`
  (interior-mutability precedent: `VadState` in `pipecrab-vad/src/stage.rs`).

Dependencies are the workspace's first target-cfg split, keeping this crate
green in the CI wasm gate:

```toml
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
silero-vad-ort = { workspace = true }
[target.'cfg(target_arch = "wasm32")'.dependencies]
silero-vad-web = { workspace = true }
```

### 4.5 `pipecrab-vad` additions: frame aggregation in `VadStage`

Engines with a fixed model window need exact frames; nothing else in the
pipeline does (Moonshine consumes utterance-scale audio gated by the
speech edges; TTS consumes text; sinks are size-agnostic — and the size is
rate-dependent even within Silero, 512 vs 256). So framing is VAD-private and
lives in `VadStage`, **not** in a separate rechunk stage: `VadStage` is a tap
that forwards audio unchanged, and a stream-level rechunker would rewrite the
public stream for every downstream consumer to serve a stage that only taps it.

References: `crates/pipecrab-core/src/processor.rs` (`decide_data` takes
`&mut self` — "all state mutation happens here"; `Decision.effects` is a
`Vec<E>` with chainable `emit`, so one decide already emits multiple effects);
`crates/pipecrab-vad/src/stage.rs` (`Detect` effect, `VadState::observe`).

```rust
// trait addition (default keeps every existing detector working):
pub trait VoiceActivityDetector: MaybeSendSync {
    /// The exact samples-per-call this engine requires, if fixed.
    /// `Some(n)` ⇒ VadStage aggregates audio and feeds exact n-sample frames.
    /// `None` (default) ⇒ chunks pass through as they arrive (current behavior).
    fn frame_len(&self) -> Option<NonZeroUsize> { None }
    // async fn detect(…) unchanged
}

pub enum VadError {
    // existing: Engine(String), UnsupportedFormat { expected, got }
    /// The detector requires fixed-size frames and got another length.
    InvalidFrameLen { expected: usize, got: usize },
}
```

`VadStage` behavior when `frame_len()` is `Some(n)`:

- `decide_data` (sync, `&mut self`, uninterruptible — the sanctioned home for
  state) forwards the audio chunk unchanged (tap preserved) while appending its
  samples to an internal buffer, then emits one `Detect` effect per completed
  `n`-sample frame — zero effects for a small chunk, several for a large one.
- The buffer is format-tagged; a format change clears it (the next `detect`
  call would reject the mismatch anyway).
- `perform` is unchanged: one `detect` per exact frame, one
  `VadState::observe` per verdict — so `start_windows`/`stop_windows` now
  count fixed 32 ms frames on every machine, instead of driver-sized chunks
  (`stop_windows: 8` = 256 ms everywhere).
- The partial-frame remainder simply waits for more samples (the reference
  never pads), and survives interrupts like any other `decide` state.

**Why `pipecrab-vad` and not `pipecrab-vad-silero`?** The trait crate's scope
is crisp: *everything about VAD that isn't the model* — the capability
contract, the edge policy, and the mock. The model crate's scope is equally
crisp: adapt Silero to that contract. Three concrete reasons the division
earns its keep (none of them hypothetical future engines):

- **The state discipline doesn't come from `pipecrab-vad`.** A collapsed
  "SileroVadStage" would still face pipecrab-core's `Processor` contract —
  sync `&mut` `decide`, async `&self` `perform` — so the recurrent state
  would still sit behind interior mutability and framing would still live in
  `decide`. Collapsing deletes the trait, not the constraint.
- **The trait's second implementor already exists: `ScriptedVad`**
  (`crates/pipecrab-vad/tests/vad_stage.rs`). The seam is what lets
  edge/debounce policy be tested deterministically without ONNX, and lets
  applications test their pipelines in CI with scripted verdicts instead of a
  2 MB model plus inference. Without it, the same seam reappears privately or
  debounce gets tested through real inference.
- **The framing/debounce coupling.** Framing exists so that
  `start_windows`/`stop_windows` count fixed frames, and that counter
  (`VadState::observe`) is `VadStage`'s. If the model crate buffered inside
  `detect`, one call could complete zero or several frames without the stage
  knowing — window counts revert to chunk-relative, and fixing *that* drags
  debounce into the model crate, gutting the stage adapter. `observe` itself
  consumes booleans: it is app-tuned pipeline policy, not model logic.

What *is* model-specific — which length per sample rate, the context prefix,
the state tensor — stays in the engine crates; the `frame_len()` addition is
one defaulted method and the buffer is ~30 dependency-free lines.

This is a semver-minor trait addition (defaulted method) plus one `VadError`
variant, shipped as its own compartmental PR before `pipecrab-vad-silero`.

## 5. ort / ONNX Runtime bundling strategy (explicit)

**Decision: `silero-vad-ort` never chooses how ONNX Runtime is acquired or
linked — the application does.** Concretely:

```toml
# silero-vad-ort/Cargo.toml
[dependencies]
ort = { version = "2.0.0-rc.12", default-features = false, features = ["std"] }

[features]
default = ["bundled-model"]
bundled-model = ["silero-vad/bundled-model"]
# passthroughs so apps can opt in through us if convenient:
download-binaries = ["ort/download-binaries", "ort/tls-native"]
load-dynamic = ["ort/load-dynamic"]
```

Why each part, with references (ort docs `setup/linking`, `setup/cargo-features`,
`backends/index`; `ort-sys` Cargo.toml; Cargo semver reference):

- **Caret, never `=`**: `ort-sys` declares `links = "onnxruntime"`, so two
  exact pins on different rcs are a hard build error for the whole dependency
  graph. Cargo's pre-release semantics let `"2.0.0-rc.12"` unify with later
  rcs and with 2.0.0 final. This is the wound prior art keeps open; we close it.
- **`default-features = false`**: ort's defaults include `download-binaries`
  (build-time fetch of static libs from pyke's CDN, statically linked). That is
  the right *application* default — a self-contained binary — but features are
  additive, so if the *library* enabled it (or `load-dynamic`, the
  `silero-vad-rust` mistake), every downstream app would be locked into that
  linking mode. The app picks exactly one of: `download-binaries` (static,
  self-contained, needs network at build time), `load-dynamic` (runtime dlopen,
  app ships/locates the dylib — the air-gapped/system-lib path along with
  `ORT_LIB_PATH` source builds or `pkg-config`), or `alternative-backend`.
- **`pub use ort;`**: apps configure `ort::init`/execution providers against
  the same ort our types use, and can see/align the version.
- **BYO `Session` (`OrtModel::from_session`)**: the escape hatch that makes our
  crate agnostic even to *how* the session was built.
- **Binary-size reality** (documented, not decided by us): the ONNX Runtime CPU
  build dwarfs the 2.2 MB model — roughly 15–30 MB statically linked. Apps that
  care use `load-dynamic` or a reduced-ops source build; `ort` publishes no
  minimal prebuilts.
- **MSRV consequence**: ort rc.12 requires Rust 1.88 (workspace `rust-version`
  is 1.75). `silero-vad-ort` (and native builds of `pipecrab-vad-silero`) carry
  a per-crate `rust-version = "1.88"` override rather than raising the
  workspace floor.

**On web there is no linking question**: no ONNX Runtime enters our wasm binary.
transformers.js loads its own onnxruntime-web (a pinned dev build) from CDN or
a self-hosted path (`env.backends.onnx.wasm.wasmPaths`); models come from HF
Hub or `env.localModelPath` (offline deployments fully supported).

Two web policies, **pinned**:

- **E2E Rust up to the transformers.js boundary.** The JS shim in each `-web`
  engine does model loading and tensor execution *only*; every piece of pre-
  and post-processing lives in Rust (for Silero: framing, context, state,
  segmentation — all in the `silero-vad` core). This rules out wrapper
  libraries like `kokoro-js` for the later TTS work: convenient, but it drags
  its own transformers.js version constraint (`^3.5.1`, rejects 4.x) and moves
  logic across the boundary. Engines bind `@huggingface/transformers` directly.
- **One transformers.js version app-wide**: `@huggingface/transformers`
  **4.2.0** (latest; carries onnxruntime-web 1.26). With no kokoro-js in the
  graph there is no constraint holding us to 3.x. All `-web` shims are written
  against, and documented for, this single version so an app ships exactly one
  transformers.js and one ort-wasm runtime — the universality that motivated
  the substrate choice.

## 6. Model-file delivery

- **Native default: bundled.** `silero_vad.onnx` (op16, 2,327,524 bytes) is MIT
  ("zero strings attached"), embedded via `include_bytes!` behind the default
  `bundled-model` feature in the core. Zero-download, offline, reproducible.
  Escape hatches: `from_bytes`/`from_file`/`from_session`.
- **Web default: fetched.** `onnx-community/silero-vad` from HF Hub
  (browser-cached), or self-hosted via `from_local`. Not embedded in the wasm
  binary by default (keeps the binary lean); an embed feature can come later if
  someone wants a single-file deployment.
- Variants (fp16 1.22 MiB, 16k-only op15 1.23 MiB, op18-ifless 2.71 MiB) are
  known and deliberately **out of scope for v1** — CPU fp32 op16 is the
  reference-parity path.

## 7. Correctness: parity with the reference

- Fixture tests: per-frame probabilities generated once from the Python
  `silero-vad` package (v6.2.x) for known WAVs, committed as fixtures; the core
  + each backend must match within tolerance. This is the test prior art fails
  (missing context prefix ⇒ silent divergence).
- Segmenter unit tests mirror `VADIterator` semantics (hysteresis, retroactive
  `temp_end`, pad arithmetic including the `− window` term in start/end
  emission) against scripted probability sequences.
- `reset()` resets *everything* — state tensor, context, segmenter bookkeeping,
  frame buffer — per the reference's `reset_states` (a tensor-only reset is a
  prior-art bug class).

## 8. Open questions (not blocking review)

1. **Web CI depth**: the wasm gate proves compilation; actual browser inference
   tests (wasm-pack + headless) are a follow-up.
2. **8 kHz**: the core supports it (it's nearly free); backends and the adapter
   default to 16 kHz mono. Any reason to surface 8 kHz in pipecrab?
3. **Rename mechanics**: exact yank/deprecation choreography for the published
   `silero-ort`/`silero-web` 0.1.0 scaffolds alongside the release-plz config
   update.

(Resolved during review: frame aggregation belongs in `VadStage`'s `decide`,
not in a separate rechunk stage or the detector — §4.5.)

## 9. References

Verified 2026-07-08.

- This repo: `crates/pipecrab-vad/src/{lib,stage}.rs`,
  `crates/pipecrab-core/src/frame.rs`, `crates/pipecrab-runtime/src/{maybe,offload}.rs`,
  `ARCHITECTURE.md`, `.github/workflows/ci.yml`.
- Silero VAD: https://github.com/snakers4/silero-vad (v6.2.1; MIT) —
  `src/silero_vad/utils_vad.py`, `examples/cpp/silero-vad-onnx.cpp`,
  wiki "Version history and Available Models".
- ort: https://ort.pyke.io / https://github.com/pykeio/ort — 2.0.0-rc.12
  (2026-03-05), docs `setup/linking`, `setup/cargo-features`,
  `migrating/version-mapping`, `backends/index`; `ort-sys` `links = "onnxruntime"`.
- transformers.js: https://www.npmjs.com/package/@huggingface/transformers
  (3.8.1 / 4.2.0), https://huggingface.co/docs/transformers.js/custom_usage;
  official examples
  https://github.com/huggingface/transformers.js-examples
  (`moonshine-web/src/worker.js`, `conversational-webgpu/src/worker.js`);
  model https://huggingface.co/onnx-community/silero-vad;
  kokoro-js 1.2.1 https://www.npmjs.com/package/kokoro-js (evidence that
  Kokoro runs on transformers.js — not a dependency; see §5).
- Prior art: https://github.com/nkeenan38/voice_activity_detector (0.2.1),
  https://github.com/sheldonix/silero-vad-rust (6.2.1),
  https://github.com/binarycrayon/silero-vad-rs (0.1.2),
  https://github.com/ricky0123/vad (vad-web 0.0.30),
  pyke ort-web https://ort.pyke.io/backends/web (0.2.2+1.27).
