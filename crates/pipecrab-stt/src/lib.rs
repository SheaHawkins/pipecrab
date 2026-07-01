//! pipecrab-stt: the speech-to-text seam.
//!
//! One trait — [`Transcriber`] — is the swappable STT capability: `f32` samples
//! in, text out. [`SttStage`] adapts any `Transcriber` into a pipeline
//! [`Stage`](pipecrab_runtime::Stage), turning a
//! [`DataFrame::Audio`](pipecrab_core::DataFrame::Audio) into a
//! [`DataFrame::Transcript`](pipecrab_core::DataFrame::Transcript) without the
//! pipeline above it ever naming a concrete model.
//!
//! Platform-neutral and `wasm32`-checkable: the concrete engines live elsewhere
//! (native `ort`, browser Transformers.js in a Web Worker), each behind this one
//! trait, so the seam itself carries no backend dependency and compiles for both
//! the host and `wasm32-unknown-unknown`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod stage;
mod transcriber;

pub use stage::{SttStage, Transcribe};
pub use transcriber::{SttError, Transcriber};
