//! kokoro-web: the browser (onnxruntime-web / WASM, Web Worker) engine for the
//! Kokoro text-to-speech model.
//!
//! This is the `wasm32` backend for a future pipecrab TTS model crate
//! (`pipecrab-tts-kokoro`, deferred); the native counterpart is
//! [`kokoro-ort`](https://docs.rs/kokoro-ort). Like the other engines it is
//! pipecrab-free — a standalone Kokoro runner the model crate wraps.
//!
//! Scaffold: the crate name is reserved and wired into the workspace and
//! release graph; the onnxruntime-web / Web Worker glue and synthesis code land
//! next.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
