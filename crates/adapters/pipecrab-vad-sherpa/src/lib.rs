//! Sherpa ONNX's Silero VAD behind pipecrab's edge-emitting detector trait.
//!
//! [`SherpaVad`] is a handle to a dedicated actor thread. The actor constructs,
//! exclusively owns, accesses, and drops one
//! [`sherpa_onnx::VoiceActivityDetector`]. Audio is accumulated into exact
//! 512-sample Silero windows, and state changes around each window become
//! [`VadEvent::SpeechStarted`](pipecrab_vad::VadEvent::SpeechStarted) and
//! [`VadEvent::SpeechStopped`](pipecrab_vad::VadEvent::SpeechStopped).
//!
//! Sherpa's completed segment audio is discarded. [`VadStage`](pipecrab_vad::VadStage)
//! owns pre-roll and forwards the original shared audio chunks.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod backend;
mod config;
mod worker;

pub use backend::Backend;
pub use config::{SherpaVadBuildError, SherpaVadConfig};
pub use worker::SherpaVad;
