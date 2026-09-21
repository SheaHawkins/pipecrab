//! pipecrab-vad-web: Silero VAD in the browser, behind pipecrab's raw-model tier.
//!
//! [`SileroScorer`] runs a Silero VAD ONNX model on [onnxruntime-web] and
//! implements [`SpeechScorer`](pipecrab_vad::SpeechScorer) — a probability per
//! fixed window, which is all a bare Silero build exposes. Wrap it in
//! [`Debounced`](pipecrab_vad::Debounced) to get the edge-emitting
//! [`VoiceActivityDetector`](pipecrab_vad::VoiceActivityDetector) that
//! [`VadStage`](pipecrab_vad::VadStage) drives, exactly as on native:
//!
//! ```ignore
//! let scorer = SileroScorer::load(&ort, SileroConfig::new("/models/silero_vad.onnx")).await?;
//! let stage = VadStage::new(Debounced::new(scorer));
//! ```
//!
//! # The engine comes from the page
//!
//! [`load`](SileroScorer::load) takes the `ort` namespace the page already
//! imported, rather than importing one itself. So the crate pins no CDN, no
//! bundler layout, and no execution provider: which onnxruntime-web build runs,
//! and where its own `.wasm` files come from, stays the application's call.
//!
//! # wasm only
//!
//! This crate is the browser engine adapter, so it is empty on every other
//! target; the native Silero path is `pipecrab-vad-sherpa`.
//!
//! [onnxruntime-web]: https://onnxruntime.ai/docs/tutorials/web/
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(target_arch = "wasm32")]
mod config;
#[cfg(target_arch = "wasm32")]
mod scorer;

#[cfg(target_arch = "wasm32")]
pub use config::SileroConfig;
#[cfg(target_arch = "wasm32")]
pub use scorer::SileroScorer;
