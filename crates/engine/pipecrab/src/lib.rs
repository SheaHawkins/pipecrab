//! pipecrab — sans-IO voice-agent pipelines, async bundled in.
//!
//! Re-exports the sans-IO core ([`pipecrab_core`]) and the runtime-agnostic
//! async orchestration layer ([`pipecrab_runtime`]) so downstream code depends
//! on one crate. The two have no name collisions.
pub use pipecrab_core::*;
pub use pipecrab_runtime::*;
