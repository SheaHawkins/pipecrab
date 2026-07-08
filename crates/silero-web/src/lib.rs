//! silero-web: the browser (onnxruntime-web / WASM, Web Worker) engine for the
//! Silero voice-activity-detection model.
//!
//! This is the `wasm32` backend behind
//! [`pipecrab-vad-silero`](https://docs.rs/pipecrab-vad-silero)'s
//! `VoiceActivityDetector` impl; the native counterpart is
//! [`silero-ort`](https://docs.rs/silero-ort).
//!
//! Scaffold: the crate name is reserved and wired into the workspace and
//! release graph; the onnxruntime-web / Web Worker glue and inference code land
//! next.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
