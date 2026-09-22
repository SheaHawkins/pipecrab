//! Talk to a fully local voice agent in the browser: Silero VAD, a
//! Transformers.js ASR model, a Transformers.js chat model, and Kokoro TTS.
//!
//! The same pipeline the `e2e-voice-agent` example runs on the desktop —
//! resample, gate on voice activity, transcribe, reply, speak — with the
//! browser's backends swapped in behind the same traits. See `README.md` for
//! the build.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(target_arch = "wasm32")]
mod app;
