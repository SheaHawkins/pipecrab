//! A stage's typed system and data input lanes.
//!
//! The system lane is polled first so lifecycle, error, and interrupt frames can
//! preempt queued media. Separate types prevent data frames from entering that
//! priority lane.

use futures::channel::mpsc::Receiver;
use futures::stream::StreamExt;
use pipecrab_core::{DataFrame, Direction, SystemFrame};

/// A frame received from [`Inbound::recv`].
#[derive(Debug)]
pub enum Received {
    /// A system frame and the direction it is travelling.
    Sys(Direction, SystemFrame),
    /// A data frame, implicitly travelling downstream.
    Data(DataFrame),
}

/// A stage's priority system lane and FIFO data lane.
///
/// Within a lane, frames keep FIFO order. Across lanes, `sys` always wins, so a
/// system frame is taken even when `data` is backed up.
pub struct Inbound {
    /// Priority lifecycle, interrupt, and error frames.
    pub sys: Receiver<(Direction, SystemFrame)>,
    /// Downstream data frames in FIFO order.
    pub data: Receiver<DataFrame>,
}

impl Inbound {
    /// Receive the next frame, draining the system lane before the data lane.
    ///
    /// Returns `None` after both lanes close.
    ///
    /// The system lane is polled first. A closed lane is skipped while the
    /// other remains open.
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

    /// Drains queued data, returning [`DataFrame::survives_flush`] frames in order.
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
