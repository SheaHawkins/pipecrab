//! Each stage has an Inbound mailbox with two typed lanes:
//! `sys` — the priority lane, drains first, carries `(Direction, SystemFrame)`.
//! `data` — the data lane, carries bare `DataFrame` (downstream only).
//!
//! Keeping the lanes typed prevents misrouting a media frame onto the system
//! lane and removes the per-frame is-system check from the hot path.

use futures::channel::mpsc::Receiver;
use futures::stream::StreamExt;
use pipecrab_core::{DataFrame, Direction, SystemFrame};

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
    /// [`futures::select_biased`] polls `sys` first, so a system frame preempts
    /// any data backlog deterministically. When a lane closes, its receiver
    /// (a [`FusedStream`]) yields `None`; the `loop` swallows that first `None`
    /// so the next iteration just skips the dead lane instead of treating it as
    /// shutdown. This is so the sys lane can keep draining even after the data
    /// lane shuts down — `None` is returned only once *both* lanes have closed.
    ///
    /// [`FusedStream`]: futures::stream::FusedStream
    pub async fn recv(&mut self) -> Option<Received> {
        loop {
            futures::select_biased! {
                sys = self.sys.next() => {
                    if let Some((dir, f)) = sys {
                        return Some(Received::Sys(dir, f));
                    }
                }
                data = self.data.next() => {
                    if let Some(f) = data {
                        return Some(Received::Data(f));
                    }
                }
                complete => return None,
            }
        }
    }

    /// Drain everything currently queued on the data lane. Frames where
    /// `survives_flush()` is false are dropped; survivors are returned in the
    /// order they arrived, for the caller to re-process. Does not block and does
    /// not touch the sys lane.
    pub fn flush_data(&mut self) -> Vec<DataFrame> {
        let mut kept = Vec::new();
        while let Ok(frame) = self.data.try_recv() {
            if frame.survives_flush() {
                kept.push(frame);
            }
        }
        kept
    }
}
