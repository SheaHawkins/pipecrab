//! Speech-to-text interfaces and pipeline integration.
//!
//! [`StreamingTranscriber`] accepts audio incrementally and emits [`SttEvent`]s.
//!
//! Engines implement it directly or use [`Buffered`] to adapt a one-shot
//! [`Transcriber`].
//!
//! [`SttStage`] adapts a [`StreamingTranscriber`] into a pipeline
//! [`Stage`](pipecrab_runtime::Stage) and fatally rejects format mismatches.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod stage;
mod streaming;
mod transcriber;

pub use stage::{SttEffect, SttStage};
pub use streaming::{Buffered, StreamingTranscriber, SttEvent};
pub use transcriber::{SttError, Transcriber};
