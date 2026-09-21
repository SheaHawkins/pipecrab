//! Transcribe speech from the browser's microphone with Silero VAD and a
//! Transformers.js ASR model.
//!
//! The same three-stage pipeline the `stt-sherpa` example runs on the desktop —
//! resample, gate on voice activity, transcribe — with the browser's backends
//! swapped in behind the same traits. See `README.md` for the build.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(target_arch = "wasm32")]
mod app;
