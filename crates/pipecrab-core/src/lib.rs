//! pipecrab-core: Sans-IO
#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Frame types and the [`CustomFrame`] extension trait.
pub mod frame;
/// The [`Processor`] trait: synchronous, sans-IO stage logic.
pub mod processor;

pub use frame::{AudioChunk, AudioFormat, CustomFrame, DataFrame, Direction, SystemFrame};
pub use processor::{Decision, Disposition, Processor};
