//! silero-ort: the native onnxruntime (`ort`) engine for the Silero
//! voice-activity-detection model.
//!
//! This is the host backend behind
//! [`pipecrab-vad-silero`](https://docs.rs/pipecrab-vad-silero)'s
//! `VoiceActivityDetector` impl; the browser counterpart is
//! [`silero-web`](https://docs.rs/silero-web).
//!
//! Scaffold: the crate name is reserved and wired into the workspace and
//! release graph; the `ort` dependency and inference code land next.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
