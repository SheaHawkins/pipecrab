//! pipecrab-runtime: Tokio
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod inbound;
/// Typed send surface for a stage's output channels.
pub mod outbound;
pub use inbound::{Inbound, Received};
pub use outbound::Outbound;