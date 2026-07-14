//! Sherpa ONNX streaming speech recognition behind PipeCrab's
//! [`StreamingTranscriber`](pipecrab_stt::StreamingTranscriber) protocol.
//!
//! [`SherpaStt`] is a handle to a dedicated actor thread. The actor constructs,
//! exclusively owns, accesses, and drops one
//! [`sherpa_onnx::OnlineRecognizer`] and its active
//! [`sherpa_onnx::OnlineStream`]. Each ready stream is decoded one step at a
//! time so [`StreamingTranscriber::cancel`](pipecrab_stt::StreamingTranscriber::cancel)
//! can invalidate the utterance between native inference calls.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod backend;
mod config;
mod worker;

pub use backend::Backend;
pub use config::{SherpaSttBuildError, SherpaSttConfig};
pub use worker::SherpaStt;
