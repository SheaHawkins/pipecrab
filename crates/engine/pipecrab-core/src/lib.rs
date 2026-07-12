//! Sans-I/O frame types and stage decisions.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Pipeline frames and the [`CustomFrame`] extension trait.
pub mod frame;
/// Synchronous, sans-I/O stage logic through [`Processor`].
pub mod processor;

pub use frame::{
    AudioChunk, AudioFormat, CustomFrame, DataFrame, Direction, Finality, Role, SystemFrame,
    Transcript,
};
pub use processor::{Decision, Disposition, Processor};
