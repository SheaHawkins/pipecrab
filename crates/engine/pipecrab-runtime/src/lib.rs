//! pipecrab-runtime: runtime-agnostic async orchestration built on `futures`.
//!
//! No async executor is baked in: the channels and run loop are plain
//! `futures` primitives, so the caller drives them (`block_on` natively,
//! `spawn_local` in the browser). Compiles for the host and
//! `wasm32-unknown-unknown`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Re-exported so [`maybe_async_trait!`] can reach `async_trait` through this
// crate (`$crate::async_trait::…`); users of the macro need no direct
// `async-trait` dependency. Hidden: an implementation detail of the macro, not
// public API.
#[doc(hidden)]
pub use async_trait;

pub mod inbound;
/// Target-conditional `Send`/`Sync` bounds (`Send` native, vacuous on wasm).
pub mod maybe;
/// The [`offload`](offload::offload) helper for running blocking work off the
/// orchestrator thread.
pub mod offload;
/// Typed send surface for a stage's output channels.
pub mod outbound;
/// The [`Pipeline`] builder and the per-stage preemptible run loop.
pub mod pipeline;
/// The [`Stage`] trait and its [`StageError`].
pub mod stage;
pub use inbound::{Inbound, Received};
pub use maybe::{MaybeSend, MaybeSendSync};
pub use offload::offload;
pub use outbound::Outbound;
pub use pipeline::{link, Pipeline, PipelineBuilder, PipelineEnds};
pub use stage::{Stage, StageError};
