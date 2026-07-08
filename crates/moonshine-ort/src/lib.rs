//! moonshine-ort: the native onnxruntime (`ort`) engine for the Moonshine
//! speech-to-text model.
//!
//! This is the host backend behind
//! [`pipecrab-stt-moonshine`](https://docs.rs/pipecrab-stt-moonshine)'s
//! `Transcriber` impl; the browser counterpart is
//! [`moonshine-web`](https://docs.rs/moonshine-web).
//!
//! Scaffold: the crate name is reserved and wired into the workspace and
//! release graph; the `ort` dependency and inference code land next.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
