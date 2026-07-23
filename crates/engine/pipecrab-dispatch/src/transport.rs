//! The transport capability traits: [`DispatchSource`] (receive events) and
//! [`DispatchSink`] (send commands), split so a transport that naturally divides
//! into a receive handle and a send handle need not force both onto one object.
//!
//! These are *interfaces* only. Concrete transports — a WebSocket, an HTTP long
//! poll, an in-process backend — live in later adapter crates
//! (`pipecrab-dispatch-websocket`, `pipecrab-dispatch-http`,
//! `pipecrab-dispatch-hermes`); a dedicated transport-interface crate would be
//! premature, so the traits stay here alongside the stages that consume them.

use async_trait::async_trait;
use pipecrab_core::{DispatchCommand, DispatchEvent};
use pipecrab_runtime::{MaybeSend, MaybeSendSync};

use crate::error::DispatchError;

/// The receive half of a transport: a stream of [`DispatchEvent`]s an external
/// backend produces, plus a synchronous cancellation.
///
/// # `MaybeSend`, not `MaybeSendSync`
///
/// [`next_event`](Self::next_event) takes `&mut self`, so [`DispatchIngress`]
/// *owns* the source and drives it single-threaded — the source never needs to
/// be shared across threads, so only `Send` (to migrate the pipeline task) is
/// required, not `Sync`. The [`DispatchSink`], by contrast, is shared (`&self`)
/// and so is `MaybeSendSync`.
///
/// # Cancellation-safety
///
/// [`DispatchIngress`] polls [`next_event`](Self::next_event) inside a
/// `select!` against the pipeline lanes, so the future may be dropped before it
/// resolves when another lane wakes first. An implementation must be
/// cancellation-safe: dropping an unresolved `next_event` must not lose an event
/// (back it with a channel receiver or an equivalent buffered source, as tokio's
/// `recv` is).
///
/// [`DispatchIngress`]: crate::DispatchIngress
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait DispatchSource: MaybeSend {
    /// Await the next external event. `Ok(None)` means the source closed
    /// gracefully (ingress stops polling it but keeps the pipeline running);
    /// `Err` is classified recoverable or fatal by [`DispatchError`].
    async fn next_event(&mut self) -> Result<Option<DispatchEvent>, DispatchError>;

    /// Synchronous, non-blocking, idempotent cancellation.
    ///
    /// A *control call* (see [`Processor`](pipecrab_core::Processor)): it flips
    /// the source's own stop signal and never blocks, so ingress can invoke it
    /// as it terminates.
    fn cancel(&self);
}

/// The send half of a transport: publishes a [`DispatchCommand`] to an external
/// backend.
///
/// `MaybeSendSync` — [`send_command`](Self::send_command) takes `&self`, so the
/// sink is a shared handle the [`DispatchEgress`](crate::DispatchEgress) borrows
/// immutably while the run loop borrows the stage.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait DispatchSink: MaybeSendSync {
    /// Publish `command` to the transport. `Err` is classified recoverable or
    /// fatal by [`DispatchError`].
    async fn send_command(&self, command: DispatchCommand) -> Result<(), DispatchError>;
}

/// Convenience marker for a transport that is both a [`DispatchSource`] and a
/// [`DispatchSink`]. Nothing *requires* one object to be both — a transport may
/// hand back a source handle and a sink handle separately — but a unified
/// transport can implement this to be named by one bound.
pub trait DispatchTransport: DispatchSource + DispatchSink {}

impl<T: DispatchSource + DispatchSink> DispatchTransport for T {}
