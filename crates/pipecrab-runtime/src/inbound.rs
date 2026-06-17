//! Each Stage has an Inbound mailbox with two lanes:
//! sys: the **priority lane**. Drains first.
//! data: the general frame lane.
//!
//! Both lanes carry `(Direction, Frame)`: direction is a tag,
//! not a separate queue, which is what lets a single `data` lane serve both
//! travel directions and what lets the `sys` lane carry an `Error` *upstream*
//! while carrying an `Interrupt` *downstream*.

use pipecrab_core::{Direction, Frame};
use tokio::sync::mpsc::Receiver;

/// The receive surface of a stage: a preempting system lane and the data lane.
///
/// Within a lane, frames keep FIFO order. Across lanes, `sys` always wins, so a
/// system frame is taken even when `data` is backed up.
///
/// Both lanes carry `(Direction, Frame)`. The data tier keeps a single lane for
/// both directions (direction is a tag, matching the prior art).
pub struct Inbound {
    /// System-tier frames (lifecycle, interruption, errors). Drained first.
    /// `Error` rides this lane *upstream*; `Interrupt`/`Start`/`Stop` ride it
    /// downstream. Sparse and latency-critical.
    pub sys: Receiver<(Direction, Frame)>,
    /// Data-tier frames (media, transcripts), in FIFO order, either direction.
    pub data: Receiver<(Direction, Frame)>,
}

impl Inbound {
    /// Receive the next frame, draining the system lane before the data lane.
    ///
    /// Returns the frame with the [`Direction`] it is travelling, or `None` once
    /// *both* lanes are closed — the run-loop's shutdown signal.
    ///
    /// The `biased` keyword polls `sys` first, so a system
    /// frame preempts any data backlog deterministically.
    pub async fn recv(&mut self) -> Option<(Direction, Frame)> {
        tokio::select! {
            biased;
            Some(df) = self.sys.recv()  => Some(df),
            Some(df) = self.data.recv() => Some(df),
            else => None,
        }
    }
}