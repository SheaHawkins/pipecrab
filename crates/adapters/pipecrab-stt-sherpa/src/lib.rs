//! Sherpa ONNX online and offline speech recognition behind PipeCrab's
//! [`StreamingTranscriber`](pipecrab_stt::StreamingTranscriber) protocol.
//!
//! [`OnlineSherpaStt`] owns Sherpa's true streaming recognizer, while
//! [`OfflineSherpaStt`] accumulates one VAD-bounded utterance for models such as
//! Moonshine v2. [`SherpaStt`] is the online default.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod backend;
mod config;
mod offline_backend;
mod offline_worker;
mod worker;

pub use backend::OnlineBackend;
/// A concise alias for the online backend boundary.
pub use backend::OnlineBackend as Backend;
pub use config::{MoonshineV2Config, OnlineSherpaSttConfig, SherpaSttBuildError, SherpaSttConfig};
pub use offline_backend::OfflineBackend;
pub use offline_worker::OfflineSherpaStt;
pub use worker::OnlineSherpaStt;
/// The default Sherpa STT implementation, backed by `OnlineRecognizer`.
pub type SherpaStt = OnlineSherpaStt;
