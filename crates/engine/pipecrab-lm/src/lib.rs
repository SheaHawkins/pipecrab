//! Language-model interfaces and pipeline integration.
//!
//! [`LanguageModel`] streams generated text from a [`Conversation`].
//!
//! [`LmStage`] tracks a conversation and turns final user
//! [`Transcript`](pipecrab_core::Transcript)s into streamed agent transcripts.
//!
//! [`ChatRole`] includes the system role absent from core's transcript role.
//! Generation may use an opaque [`GenParams::grammar`] constraint.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod model;
mod stage;

pub use model::{
    ChatRole, Conversation, GenParams, LanguageModel, LmError, Message, TokenOut, TokenStream,
};
pub use stage::{Generate, LmStage};
