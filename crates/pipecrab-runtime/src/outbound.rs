use pipecrab_core::{DataFrame, Direction, SystemFrame};
use tokio::sync::mpsc::{error::SendError, Sender};

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
    pub async fn send_data(&self, frame: DataFrame) -> Result<(), SendError<DataFrame>> {
        self.data.send(frame).await
    }

    /// Send a system frame in the given direction.
    pub async fn send_system(
        &self,
        dir: Direction,
        frame: SystemFrame,
    ) -> Result<(), SendError<(Direction, SystemFrame)>> {
        self.sys.send((dir, frame)).await
    }
}
