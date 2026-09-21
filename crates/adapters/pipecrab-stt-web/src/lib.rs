//! pipecrab-stt-web: browser speech-to-text for pipecrab, over Transformers.js.
//!
//! [`TransformersStt`] runs a Transformers.js `automatic-speech-recognition`
//! pipeline — Whisper, Moonshine, anything with an ONNX export on the Hub — and
//! implements the one-shot [`Transcriber`](pipecrab_stt::Transcriber): samples
//! in, a transcript out, no partials. Wrap it in
//! [`Buffered`](pipecrab_stt::Buffered) to reach the streaming interface
//! [`SttStage`](pipecrab_stt::SttStage) drives, exactly as the native Moonshine
//! adapter does:
//!
//! ```ignore
//! let stt = TransformersStt::load(&transformers, TransformersSttConfig::new(model)).await?;
//! let stage = SttStage::new(Buffered::new(stt));
//! ```
//!
//! # The engine comes from the page
//!
//! [`load`](TransformersStt::load) takes the Transformers.js namespace the page
//! already imported, rather than importing one itself. So the crate pins no CDN
//! and no bundler layout, and where the engine puts its own work — a Web Worker,
//! WebGPU, the main thread — stays the application's call.
//!
//! # wasm only
//!
//! This crate is the browser engine adapter, so it is empty on every other
//! target; the native path is `pipecrab-stt-sherpa`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(target_arch = "wasm32")]
mod config;
#[cfg(target_arch = "wasm32")]
mod transcriber;

#[cfg(target_arch = "wasm32")]
pub use config::TransformersSttConfig;
#[cfg(target_arch = "wasm32")]
pub use transcriber::TransformersStt;
