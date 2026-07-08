//! pipecrab-vad-silero: the Silero VAD, wired into pipecrab's
//! [`VoiceActivityDetector`] seam.
//!
//! Silero is a small, fast voice-activity model. This crate implements
//! [`pipecrab_vad::VoiceActivityDetector`] on top of a Silero engine, picking
//! the backend by target: the native `ort`-hosted engine ([`silero-ort`]) on
//! the host, and the browser onnxruntime-web engine ([`silero-web`]) on
//! `wasm32`. The pipeline above depends only on the seam, never on this crate.
//!
//! Scaffold: the crate and its place in the workspace and release graph are set
//! up, and the seam types are re-exported for convenience; the concrete
//! `VoiceActivityDetector` impl lands with the engine crates.
//!
//! [`silero-ort`]: https://docs.rs/silero-ort
//! [`silero-web`]: https://docs.rs/silero-web
#![forbid(unsafe_code)]

#[doc(no_inline)]
pub use pipecrab_vad::{VadError, VadVerdict, VoiceActivityDetector};
