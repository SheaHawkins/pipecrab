use futures::channel::mpsc::{SendError, Sender};
use futures::sink::SinkExt;
use pipecrab_core::{DataFrame, Direction, SystemFrame};

/// The send surface of a stage: typed sends for the data and system lanes.
///
/// `send_data` targets the downstream data lane; `send_system` targets the
/// system lane with an explicit [`Direction`].
pub struct Outbound {
    /// Downstream data channel.
    pub data: Sender<DataFrame>,
    /// Bidirectional system channel.
    pub sys: Sender<(Direction, SystemFrame)>,
}

impl Outbound {
    /// Send a data frame downstream.
    ///
    /// Takes `&self` (not `&mut self`) so a stage can send while it is borrowed
    /// immutably by the run loop. `futures`' `Sink::send` needs `&mut`, so we
    /// send on a cheap clone of the shared sender; clones feed the same channel.
    pub async fn send_data(&self, frame: DataFrame) -> Result<(), SendError> {
        self.data.clone().send(frame).await
    }

    /// Send a system frame in the given direction. Takes `&self` for the same
    /// reason as [`send_data`](Self::send_data).
    pub async fn send_system(&self, dir: Direction, frame: SystemFrame) -> Result<(), SendError> {
        self.sys.clone().send((dir, frame)).await
    }
}
