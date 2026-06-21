//! Each Stage has an Inbound mailbox with two lanes:
//! sys: the **priority lane**. Drains first.
//! data: the general frame lane (downstream only; carries no Direction).
//!
//! The sys lane carries `(Direction, Frame)`, letting it transport an `Error`
//! upstream or an `Interrupt` downstream. The data lane carries only `Frame`
//! because media flows in one direction: source → sink.

use pipecrab_core::{Direction, Frame};
use tokio::sync::mpsc::Receiver;

/// The receive surface of a stage: a preempting system lane and the data lane.
///
/// Within a lane, frames keep FIFO order. Across lanes, `sys` always wins, so a
/// system frame is taken even when `data` is backed up.
///
/// `sys` carries `(Direction, Frame)`; `data` carries bare `Frame` because
/// media is downstream-only — callers that need to know the direction of a
/// data frame can assume [`Direction::Down`].
pub struct Inbound {
    /// System-tier frames (lifecycle, interruption, errors). Drained first.
    /// `Error` rides this lane *upstream*; `Interrupt`/`Start`/`Stop` ride it
    /// downstream. Sparse and latency-critical.
    pub sys: Receiver<(Direction, Frame)>,
    /// Data-tier frames (media, transcripts), in FIFO order, downstream only.
    pub data: Receiver<Frame>,
}

impl Inbound {
    /// Receive the next frame, draining the system lane before the data lane.
    ///
    /// Returns `(direction, frame)`, or `None` once *both* lanes are closed —
    /// the run-loop's shutdown signal. Data frames synthesise [`Direction::Down`].
    ///
    /// The `biased` keyword polls `sys` first, so a system
    /// frame preempts any data backlog deterministically.
    pub async fn recv(&mut self) -> Option<(Direction, Frame)> {
        tokio::select! {
            biased;
            Some(df) = self.sys.recv()  => Some(df),
            Some(f)  = self.data.recv() => Some((Direction::Down, f)),
            else => None,
        }
    }
}