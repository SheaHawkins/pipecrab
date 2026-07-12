use futures::channel::mpsc::{SendError, Sender};
use futures::sink::SinkExt;
use pipecrab_core::{DataFrame, Direction, SystemFrame};

/// A stage's typed data and system output lanes.
pub struct Outbound {
    /// Downstream data channel.
    pub data: Sender<DataFrame>,
    /// Bidirectional system channel.
    pub sys: Sender<(Direction, SystemFrame)>,
}

impl Outbound {
    /// Send a data frame downstream.
    ///
    /// Clones the shared sender so it can be called through `&self`.
    pub async fn send_data(&self, frame: DataFrame) -> Result<(), SendError> {
        self.data.clone().send(frame).await
    }

    /// Sends a system frame in the given direction.
    pub async fn send_system(&self, dir: Direction, frame: SystemFrame) -> Result<(), SendError> {
        self.sys.clone().send((dir, frame)).await
    }
}
