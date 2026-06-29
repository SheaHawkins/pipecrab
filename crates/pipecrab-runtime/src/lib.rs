//! pipecrab-runtime: runtime-agnostic async orchestration built on `futures`.
//!
//! No async executor is baked in: the channels and run loop are plain
//! `futures` primitives, so the caller drives them (`block_on` natively,
//! `spawn_local` in the browser). Compiles for the host and
//! `wasm32-unknown-unknown`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod inbound;
/// Typed send surface for a stage's output channels.
pub mod outbound;
pub use inbound::{Inbound, Received};
pub use outbound::Outbound;