//! Sherpa ONNX Kokoro text-to-speech behind pipecrab's
//! [`Synthesizer`](pipecrab_tts::Synthesizer) protocol.
//!
//! [`SherpaTts`] owns Sherpa's offline TTS engine on a dedicated actor thread
//! and streams each generated sentence to the caller as it is produced, so
//! playback of a long reply starts after its first sentence and a barge-in
//! [`cancel`](pipecrab_tts::Synthesizer::cancel) stops the engine within one
//! sentence. [`KokoroConfig`] names the Kokoro model files.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod backend;
mod config;
mod worker;

pub use backend::{Backend, Emit};
pub use config::{KokoroConfig, SherpaTtsBuildError};
pub use worker::SherpaTts;
