//! Native llama.cpp implementation of Pipecrab's language-model capability.
//!
//! [`LlamaCpp`] is a lightweight, cloneable handle. A dedicated worker thread
//! owns llama.cpp's model and context for their entire lifetime, streams decoded
//! text as [`ModelDelta::Text`](pipecrab_lm::ModelDelta::Text) through Pipecrab's
//! [`ModelStream`](pipecrab_lm::ModelStream), and checks an atomic cancellation
//! flag between tokens. The worker arrangement keeps native inference state off
//! the pipeline thread and makes barge-in bounded by one decode step.
//!
//! # Tool calling
//!
//! `llama-cpp-2` exposes only the legacy role/content chat-template call, so a
//! [`ToolDialect`] owns every model-specific string: it renders declarations
//! into the system message, converts the tool schemas to a GBNF, and reads a
//! call back out of the token stream. The default is [`ChatMlXml`].
//!
//! The grammar is attached *lazily*, triggered by the dialect's open delimiter:
//! until a call begins, sampling is unconstrained and an ordinary reply streams
//! as it would with no tools at all. Once a call begins, its text is withheld
//! from the stream and surfaces as one
//! [`ModelDelta::ToolCall`](pipecrab_lm::ModelDelta::ToolCall) — a fragment that
//! escaped would be spoken aloud by the TTS stage. Because the grammar's entry
//! rule is one call, a turn yields at most one, and no text follows it.
//!
//! Passing no tools disables all of it: the prompt and the delta stream are
//! exactly what a dialect-free adapter produces.
//!
//! The crate builds llama.cpp with Metal on iOS and the optimized ARM CPU backend
//! on Android. Android uses the shared C++ runtime expected by an NDK application;
//! callers can request GPU layers in [`LlamaCppConfig`], but should only do so
//! after a device-specific startup smoke test.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod config;
mod dialect;
mod intercept;
mod worker;

pub use config::{LlamaCppBuildError, LlamaCppConfig};
pub use dialect::{ChatMlXml, TOOL_CALL_ROOT, ToolDialect};
pub use worker::LlamaCpp;
