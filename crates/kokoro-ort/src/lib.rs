//! kokoro-ort: the native onnxruntime (`ort`) engine for the Kokoro
//! text-to-speech model.
//!
//! This is the host backend for a future pipecrab TTS model crate
//! (`pipecrab-tts-kokoro`, deferred); the browser counterpart is
//! [`kokoro-web`](https://docs.rs/kokoro-web). Like the other engines it is
//! pipecrab-free — a standalone Kokoro runner the model crate wraps.
//!
//! Scaffold: the crate name is reserved and wired into the workspace and
//! release graph; the `ort` dependency and synthesis code land next.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
