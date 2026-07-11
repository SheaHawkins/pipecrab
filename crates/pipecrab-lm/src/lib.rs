//! pipecrab-lm: the language-model interface.
//!
//! [`LanguageModel`] is the swappable LM capability the conversation loop drives:
//! a [`Conversation`] in, generated text out *incrementally* — every
//! [`TokenStream`] item is a preemption point, so a barge-in
//! [`Interrupt`](pipecrab_core::SystemFrame::Interrupt) stops the reply within a
//! single delta. Concrete engines stay behind it, so the pipeline never names
//! one.
//!
//! [`LmStage`] adapts any [`LanguageModel`] into a pipeline
//! [`Stage`](pipecrab_runtime::Stage): it tracks the running [`Conversation`]
//! (system prompt injected at construction), and on a final user
//! [`Transcript`](pipecrab_core::Transcript) it appends the turn and streams a
//! generated reply back as agent transcripts — partials as deltas arrive, then a
//! final.
//!
//! The chat-context types ([`ChatRole`], [`Message`], [`Conversation`]) are the
//! LM's own view of the dialogue, kept distinct from core's transcript
//! [`Role`](pipecrab_core::Role) because the LM needs a
//! [`System`](ChatRole::System) role the transcript stream has no notion of. The
//! trait carries an optional [`grammar`](GenParams::grammar) constraint but no
//! tool or dispatch concept — a dispatcher parses constrained output *above* this
//! layer.
//!
//! Platform-neutral and `wasm32`-checkable: the concrete engines live elsewhere
//! (a native `llama.cpp` context, a browser engine in a Web Worker), each behind
//! this trait, so the interface itself carries no backend dependency and compiles
//! for both the host and `wasm32-unknown-unknown`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod model;
mod stage;

pub use model::{
    ChatRole, Conversation, GenParams, LanguageModel, LmError, Message, TokenOut, TokenStream,
};
pub use stage::{Generate, LmStage};
