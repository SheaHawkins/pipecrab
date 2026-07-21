//! pipecrab-core: Sans-IO
#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Frame types and the [`CustomFrame`] extension trait.
pub mod frame;
/// The [`Processor`] trait: synchronous, sans-IO stage logic.
pub mod processor;

pub use frame::{
    AudioChunk, AudioFormat, CustomFrame, DataFrame, Direction, DispatchCommand, DispatchEvent,
    DispatchFrame, Finality, ModelFrame, ModelInput, ModelMessage, Role, SystemFrame, ToolCall,
    Transcript,
};
pub use processor::{Decision, Disposition, Processor};
