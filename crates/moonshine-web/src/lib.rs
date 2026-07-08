//! moonshine-web: the browser (onnxruntime-web / WASM, Web Worker) engine for
//! the Moonshine speech-to-text model.
//!
//! This is the `wasm32` backend behind
//! [`pipecrab-stt-moonshine`](https://docs.rs/pipecrab-stt-moonshine)'s
//! `Transcriber` impl; the native counterpart is
//! [`moonshine-ort`](https://docs.rs/moonshine-ort).
//!
//! Scaffold: the crate name is reserved and wired into the workspace and
//! release graph; the onnxruntime-web / Web Worker glue and inference code land
//! next.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
