//! pipecrab-lm: the provider-neutral structured-generation interface.
//!
//! [`LanguageModel`] is the swappable LM capability the conversation loop drives:
//! a [`Conversation`] in, a [`ModelStream`] of [`ModelDelta`]s out
//! *incrementally* — text to append, or a complete [`ToolCall`]. Every item is a
//! preemption point, so a barge-in
//! [`Interrupt`](pipecrab_core::SystemFrame::Interrupt) stops the reply within a
//! single delta. Concrete engines stay behind the trait, so the pipeline never
//! names one.
//!
//! # `ModelDelta` vs [`ModelFrame`](pipecrab_core::ModelFrame)
//!
//! A [`ModelDelta`] is the *internal* protocol between a [`LanguageModel`] and
//! [`LmStage`], not a pipeline frame. The stage translates deltas into native
//! frames: visible text becomes agent [`Transcript`](pipecrab_core::Transcript)s
//! (cumulative partials, then a final) because a transcript is prose; a tool call
//! becomes [`ModelFrame::ToolCall`](pipecrab_core::ModelFrame::ToolCall) because
//! it is structured protocol; the lifecycle becomes
//! [`GenerationStarted`](pipecrab_core::ModelFrame::GenerationStarted) /
//! [`GenerationFinished`](pipecrab_core::ModelFrame::GenerationFinished).
//! Tool-call syntax, JSON, and provider metadata never enter a transcript.
//!
//! # Tools
//!
//! [`ToolDefinition::parameters`] is a [`serde_json::Value`] so a framework's
//! schema reaches a hosted adapter without a string round trip. Core stays
//! JSON-free: a [`ToolCall`] carries validated arguments as JSON *text*, and
//! [`ModelDelta::tool_call`] normalizes a JSON object into it.
//! Provider-specific streaming, tool-call fragments, and constrained output are
//! parsed *inside* the implementation; the stage sees only complete calls.
//! [`LmStage::with_tools`] / [`add_tools`](LmStage::add_tools) configure tools on
//! the stage — validated (duplicate names rejected) and passed to every
//! generation. An adapter that wraps a higher-level agent (e.g. Rig) keeps its own
//! registered tools internal, so the stage neither reads nor copies them.
//!
//! The chat-context types ([`Message`], [`Conversation`]) preserve structured
//! assistant turns, tool calls, tool results, and external events so a hosted
//! adapter can reconstruct valid provider history.
//!
//! Platform-neutral and `wasm32`-checkable: the concrete engines live elsewhere
//! (a native `llama.cpp` context, a hosted Rig agent, a browser Worker), each
//! behind this trait, so the interface itself carries no backend dependency and
//! compiles for both the host and `wasm32-unknown-unknown`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod model;
mod stage;

pub use model::{
    Conversation, GenParams, LanguageModel, LmConfigError, LmError, Message, ModelDelta,
    ModelStream, ToolDefinition,
};
pub use pipecrab_core::ToolCall;
pub use stage::{Generate, LmStage};
