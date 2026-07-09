//! silero-vad-web: the browser (transformers.js in a Web Worker) engine for
//! Silero VAD.
//!
//! This is the `wasm32` backend behind
//! [`pipecrab-vad-silero`](https://docs.rs/pipecrab-vad-silero)'s
//! `VoiceActivityDetector` impl; the native counterpart is
//! [`silero-vad-ort`](https://docs.rs/silero-vad-ort). It supplies the async
//! inference call for the shared [`silero-vad`](https://docs.rs/silero-vad)
//! core; the model logic — context prefix, state lifecycle, segmentation —
//! lives in the core, so web and native are numerically identical.
//!
//! # Design (locked in `docs/plans/silero-vad.md`, §4.3 and §5)
//!
//! This crate is **unimplemented scaffolding**; the decisions below are recorded
//! so implementation doesn't drift.
//!
//! ## Substrate: transformers.js, not raw onnxruntime-web
//!
//! Inference goes through `@huggingface/transformers` **4.2.0** (which carries
//! onnxruntime-web 1.26) — the *same* substrate and version as `moonshine-web`
//! and `kokoro-web`, so an app ships exactly one transformers.js and one
//! ort-wasm runtime underneath it. A thin JS shim (`js/silero.js`) does model
//! loading and tensor execution *only*; Rust binds it with
//! `#[wasm_bindgen(module = "/js/silero.js")]` +
//! `wasm_bindgen_futures::JsFuture`. Everything else stays in Rust (framing,
//! context, state, segmentation — all in the core): the pipeline is Rust
//! end-to-end up to the transformers.js boundary. This proven pattern is what
//! Hugging Face's official `conversational-webgpu` / `moonshine-web` examples
//! use (`AutoModel` + `config: { model_type: "custom" }` with manual `state`
//! feedback).
//!
//! ## Planned surface
//!
//! - `WebModel` (a `JsValue` handle to the loaded `AutoModel`) with `from_hub()`
//!   (fetch `onnx-community/silero-vad`, browser-cached) and
//!   `from_local(base_url)` (self-hosted / offline via transformers.js
//!   `env.localModelPath`) constructors, plus an async
//!   `infer(input, state, sr) -> f32`.
//! - `SileroVadWeb` — the async engine mirroring `silero_vad::SileroVad`, built
//!   on `SileroSession`.
//!
//! ## Locked notes
//!
//! - **We feed the context-prefixed `[1, 576]` input on web too.** The official
//!   JS examples feed a bare `[1, 512]` window and so diverge numerically from
//!   the Python reference; the graph accepts both lengths, and the Python
//!   wrapper feeds 576 to this same graph. The core makes web and native
//!   identical.
//! - Per-chunk interop cost is negligible: ~2 KB crossing Rust→JS every 32 ms,
//!   ~µs of interop against ~1 ms of inference — the official demos already
//!   cross worklet→worker per chunk.
//! - Model files come from HF Hub by default (browser-cached) or a self-hosted
//!   path; not embedded in the wasm binary, keeping it lean.
//! - This crate is in the CI wasm gate (`.github/workflows/ci.yml`), like the
//!   `silero-web` name it replaces.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
