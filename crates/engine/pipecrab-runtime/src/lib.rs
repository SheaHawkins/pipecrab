//! Runtime-agnostic pipeline orchestration built on `futures`.
//!
//! The caller supplies the executor. Native and `wasm32-unknown-unknown`
//! targets are supported.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Re-exported so [`maybe_async_trait!`] can reach `async_trait` through this
// crate (`$crate::async_trait::…`); users of the macro need no direct
// `async-trait` dependency. Hidden: an implementation detail of the macro, not
// public API.
#[doc(hidden)]
pub use async_trait;

pub mod inbound;
/// Target-conditional `Send` and `Sync` bounds.
pub mod maybe;
/// The [`offload`](offload::offload) helper for running blocking work off the
/// orchestrator thread.
pub mod offload;
/// A stage's typed output channels.
pub mod outbound;
/// Pipeline construction and execution.
pub mod pipeline;
/// The [`Stage`] trait and its [`StageError`].
pub mod stage;
pub use inbound::{Inbound, Received};
pub use maybe::{MaybeSend, MaybeSendSync};
pub use offload::offload;
pub use outbound::Outbound;
pub use pipeline::{link, Pipeline, PipelineBuilder, PipelineEnds};
pub use stage::{Stage, StageError};
