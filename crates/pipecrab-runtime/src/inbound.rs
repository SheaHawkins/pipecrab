//! Each stage has an Inbound mailbox with two typed lanes:
//! `sys` — the priority lane, drains first, carries `(Direction, SystemFrame)`.
//! `data` — the data lane, carries bare `DataFrame` (downstream only).
//!
//! Keeping the lanes typed prevents misrouting a media frame onto the system
//! lane and removes the per-frame is-system check from the hot path.

use pipecrab_core::{DataFrame, Direction, SystemFrame};
use tokio::sync::mpsc::Receiver;

/// A frame received from [`Inbound::recv`]: either a system frame (with its
/// travel direction) or a data frame (always downstream).
#[derive(Debug)]
pub enum Received {
    /// A system frame and the direction it is travelling.
    Sys(Direction, SystemFrame),
    /// A data frame, implicitly travelling downstream.
    Data(DataFrame),
}

/// The receive surface of a stage: a preempting system lane and the data lane.
///
/// Within a lane, frames keep FIFO order. Across lanes, `sys` always wins, so a
/// system frame is taken even when `data` is backed up.
pub struct Inbound {
    /// System-tier frames (lifecycle, interruption, errors). Drained first.
    /// `Error` rides this lane *upstream*; `Interrupt`/`Start`/`Stop` ride it
    /// downstream. Sparse and latency-critical.
    pub sys: Receiver<(Direction, SystemFrame)>,
    /// Data-tier frames (media, transcripts), in FIFO order, downstream only.
    pub data: Receiver<DataFrame>,
}

impl Inbound {
    /// Receive the next frame, draining the system lane before the data lane.
    ///
    /// Returns [`Received::Sys`] or [`Received::Data`], or `None` once *both*
    /// lanes are closed — the run-loop's shutdown signal.
    ///
    /// The `biased` keyword polls `sys` first so a system frame preempts any
    /// data backlog deterministically.
    pub async fn recv(&mut self) -> Option<Received> {
        tokio::select! {
            biased;
            Some((dir, f)) = self.sys.recv()  => Some(Received::Sys(dir, f)),
            Some(f)        = self.data.recv() => Some(Received::Data(f)),
            else => None,
        }
    }
}
