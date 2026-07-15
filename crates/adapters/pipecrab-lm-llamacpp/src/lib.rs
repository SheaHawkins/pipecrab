//! Native llama.cpp implementation of Pipecrab's language-model capability.
//!
//! [`LlamaCpp`] is a lightweight, cloneable handle. A dedicated worker thread
//! owns llama.cpp's model and context for their entire lifetime, streams decoded
//! text through Pipecrab's [`TokenStream`](pipecrab_lm::TokenStream), and checks
//! an atomic cancellation flag between tokens. The worker arrangement keeps
//! native inference state off the pipeline thread and makes barge-in bounded by
//! one decode step.
//!
//! The crate builds llama.cpp with Metal on iOS and the optimized ARM CPU backend
//! on Android. Android uses the shared C++ runtime expected by an NDK application;
//! callers can request GPU layers in [`LlamaCppConfig`], but should only do so
//! after a device-specific startup smoke test.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod config;
mod worker;

pub use config::{LlamaCppBuildError, LlamaCppConfig};
pub use worker::LlamaCpp;
