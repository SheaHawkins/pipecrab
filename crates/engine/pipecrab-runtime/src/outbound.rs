use std::sync::atomic::{AtomicU64, Ordering};

use futures::channel::mpsc::{SendError, Sender};
use futures::sink::SinkExt;
use pipecrab_core::{DataFrame, Direction, SystemFrame};

use crate::inbound::Stamped;

/// The send surface of a stage: typed sends for the data and system lanes.
///
/// `send_data` targets the downstream data lane; `send_system` targets the
/// system lane with an explicit [`Direction`]. Every send stamps its frame from
/// one per-link counter shared by both lanes, which is what lets the receiving
/// side's flush distinguish frames queued before an `Interrupt` from frames
/// queued after it.
///
/// Constructed only by [`link`](crate::link).
pub struct Outbound {
    /// Downstream data channel.
    pub(crate) data: Sender<Stamped<DataFrame>>,
    /// Bidirectional system channel.
    pub(crate) sys: Sender<Stamped<(Direction, SystemFrame)>>,
    /// Per-link monotonic stamp shared by both lanes. Atomic so sends work
    /// through `&self`; sends on one `Outbound` are not otherwise synchronised,
    /// so issue them sequentially.
    pub(crate) seq: AtomicU64,
}

impl Outbound {
    /// Send a data frame downstream.
    ///
    /// Takes `&self` (not `&mut self`) so a stage can send while it is borrowed
    /// immutably by the run loop. `futures`' `Sink::send` needs `&mut`, so we
    /// send on a cheap clone of the shared sender; clones feed the same channel.
    pub async fn send_data(&self, frame: DataFrame) -> Result<(), SendError> {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.data.clone().send(Stamped { seq, frame }).await
    }

    /// Send a system frame in the given direction. Takes `&self` for the same
    /// reason as [`send_data`](Self::send_data).
    pub async fn send_system(&self, dir: Direction, frame: SystemFrame) -> Result<(), SendError> {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.sys
            .clone()
            .send(Stamped {
                seq,
                frame: (dir, frame),
            })
            .await
    }
}
