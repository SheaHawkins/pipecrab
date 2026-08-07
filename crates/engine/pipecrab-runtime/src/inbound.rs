//! Each stage has an Inbound mailbox with two typed lanes:
//! `sys` — the priority lane, drains first, carries `(Direction, SystemFrame)`.
//! `data` — the data lane, carries bare `DataFrame` (downstream only).
//!
//! Keeping the lanes typed prevents misrouting a media frame onto the system
//! lane and removes the per-frame is-system check from the hot path.
//!
//! Every frame crosses its link carrying a per-link sequence stamp (see
//! [`Stamped`]), which makes the interrupt flush *causal*: a flush drops only
//! frames queued before the system frame it flushes against, so a barge-in
//! utterance sent behind its own `Interrupt` is never destroyed by it.

use futures::channel::mpsc::Receiver;
use futures::stream::StreamExt;
use pipecrab_core::{DataFrame, Direction, SystemFrame};

/// A frame paired with the sequence stamp its link's
/// [`Outbound`](crate::Outbound) applied.
///
/// Both lanes of one link share a single monotonic counter, so stamps order
/// frames *across* lanes: [`Inbound::flush_data`] keeps any data frame stamped
/// at or after the system frame being flushed against.
#[derive(Debug)]
pub(crate) struct Stamped<T> {
    /// Per-link monotonic sequence number; `0` is never issued.
    pub(crate) seq: u64,
    /// The carried frame.
    pub(crate) frame: T,
}

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
///
/// Constructed only by [`link`](crate::link); the lanes are private so every
/// receive goes through [`recv`](Self::recv), which maintains the flush floor
/// [`flush_data`](Self::flush_data) relies on.
pub struct Inbound {
    /// System-tier frames (lifecycle, interruption, errors). Drained first.
    /// `Error` rides this lane *upstream*; `Interrupt`/`Start`/`Stop` ride it
    /// downstream. Sparse and latency-critical.
    pub(crate) sys: Receiver<Stamped<(Direction, SystemFrame)>>,
    /// Data-tier frames (media, transcripts), in FIFO order, downstream only.
    pub(crate) data: Receiver<Stamped<DataFrame>>,
    /// Stamp of the most recent system frame taken off `sys` — the floor
    /// [`flush_data`](Self::flush_data) flushes up to.
    pub(crate) flush_floor: u64,
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
                    if let Some(Stamped { seq, frame: (dir, f) }) = sys {
                        self.flush_floor = seq;
                        return Some(Received::Sys(dir, f));
                    }
                }
                data = self.data.next() => {
                    if let Some(Stamped { frame, .. }) = data {
                        return Some(Received::Data(frame));
                    }
                }
                complete => return None,
            }
        }
    }

    /// Drain everything currently queued on the data lane. A frame queued
    /// *before* the most recently received system frame is kept only if
    /// `survives_flush()`; a frame queued at or after it is always kept.
    /// Keepers are returned in arrival order, for the caller to re-process.
    /// Does not block and does not touch the sys lane.
    ///
    /// Only meaningful straight after receiving the system frame to flush
    /// against — receiving another system frame moves the floor.
    pub fn flush_data(&mut self) -> Vec<DataFrame> {
        self.flush_data_stamped()
            .into_iter()
            .map(|stamped| stamped.frame)
            .collect()
    }

    /// [`flush_data`](Self::flush_data), keeping the stamps: the run loop holds
    /// keepers across a later interrupt, whose flush must re-judge them by seq.
    pub(crate) fn flush_data_stamped(&mut self) -> Vec<Stamped<DataFrame>> {
        let mut kept = Vec::new();
        while let Ok(stamped) = self.data.try_recv() {
            if stamped.seq >= self.flush_floor || stamped.frame.survives_flush() {
                kept.push(stamped);
            }
        }
        kept
    }

    /// Await the next system frame, ignoring the data lane and maintaining the
    /// flush floor exactly as [`recv`](Self::recv) does.
    ///
    /// `None` once the system lane closes, and immediately so from then on — a
    /// caller racing this against other work must stop polling it at that
    /// point. Cancellation-safe: a dropped future takes nothing off the lane.
    ///
    /// This is what keeps an application's output pump preemptible while it is
    /// busy with a data frame. A stage gets sys priority from the run loop's
    /// own race; the tail lane belongs to the application, so its pump must run
    /// the same race itself (see the e2e examples' `pump_out`).
    pub async fn recv_sys(&mut self) -> Option<(Direction, SystemFrame)> {
        let Stamped {
            seq,
            frame: (dir, frame),
        } = self.sys.next().await?;
        self.flush_floor = seq;
        Some((dir, frame))
    }

    /// Take one already-queued system frame without blocking, maintaining the
    /// flush floor exactly as [`recv`](Self::recv) would. `None` when the sys
    /// lane is empty or closed. Lets the run loop keep sys priority while it
    /// replays flush keepers instead of awaiting `recv`.
    pub(crate) fn try_recv_sys(&mut self) -> Option<(Direction, SystemFrame)> {
        match self.sys.try_recv() {
            Ok(Stamped {
                seq,
                frame: (dir, frame),
            }) => {
                self.flush_floor = seq;
                Some((dir, frame))
            }
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Lane-close semantics need one lane to close while the other stays open,
    //! which the public [`link`](crate::link) surface cannot express (one
    //! `Outbound` owns both senders) — so these live here, on the raw lanes.

    use futures::FutureExt;
    use futures::channel::mpsc;
    use futures::executor::block_on;
    use pipecrab_core::Transcript;

    use super::*;

    #[allow(clippy::type_complexity)]
    fn lanes() -> (
        mpsc::Sender<Stamped<(Direction, SystemFrame)>>,
        mpsc::Sender<Stamped<DataFrame>>,
        Inbound,
    ) {
        let (sys_tx, sys) = mpsc::channel(16);
        let (data_tx, data) = mpsc::channel(16);
        (
            sys_tx,
            data_tx,
            Inbound {
                sys,
                data,
                flush_floor: 0,
            },
        )
    }

    #[test]
    fn both_lanes_closed_yields_none() {
        block_on(async {
            let (sys_tx, data_tx, mut inb) = lanes();
            drop(sys_tx);
            drop(data_tx);
            assert!(
                inb.recv().await.is_none(),
                "closed lanes must signal shutdown via None"
            );
        });
    }

    #[test]
    fn one_closed_lane_does_not_signal_shutdown() {
        block_on(async {
            let (sys_tx, data_tx, mut inb) = lanes();
            // Data lane closes while sys is still open but empty.
            drop(data_tx);
            // recv must NOT resolve to None — a single closed lane is not
            // shutdown. `now_or_never` yields `None` while still pending.
            assert!(
                inb.recv().now_or_never().is_none(),
                "a still-open sys lane must keep recv pending, not report shutdown",
            );

            // The other lane closing too is what finally yields `None`.
            drop(sys_tx);
            assert!(
                matches!(inb.recv().now_or_never(), Some(None)),
                "both lanes closed must resolve immediately to None",
            );
        });
    }

    #[test]
    fn closed_sys_lane_still_serves_buffered_data() {
        block_on(async {
            let (sys_tx, mut data_tx, mut inb) = lanes();
            data_tx
                .try_send(Stamped {
                    seq: 1,
                    frame: Transcript::user_final("after sys closed").into(),
                })
                .unwrap();
            // Sys lane closes, but a buffered data frame must still be
            // delivered.
            drop(sys_tx);

            match inb.recv().await.unwrap() {
                Received::Data(DataFrame::Transcript(s)) => {
                    assert_eq!(s.text, "after sys closed".into())
                }
                other => panic!("closed sys lane must not block the data lane, got {other:?}"),
            }
        });
    }

    #[test]
    fn recv_sys_takes_the_system_frame_past_a_backed_up_data_lane() {
        block_on(async {
            let (mut sys_tx, mut data_tx, mut inb) = lanes();
            data_tx
                .try_send(Stamped {
                    seq: 1,
                    frame: Transcript::user_final("stale").into(),
                })
                .unwrap();
            sys_tx
                .try_send(Stamped {
                    seq: 2,
                    frame: (Direction::Down, SystemFrame::Interrupt),
                })
                .unwrap();
            data_tx
                .try_send(Stamped {
                    seq: 3,
                    frame: Transcript::user_final("barge-in").into(),
                })
                .unwrap();

            assert!(
                matches!(
                    inb.recv_sys().await,
                    Some((Direction::Down, SystemFrame::Interrupt))
                ),
                "recv_sys must reach the system frame without draining data first",
            );
            // The floor moved, so the causal flush still discriminates.
            let kept = inb.flush_data();
            assert_eq!(kept.len(), 1, "only the post-Interrupt frame survives");
            match &kept[0] {
                DataFrame::Transcript(s) => assert_eq!(s.text, "barge-in".into()),
                other => panic!("wrong survivor: {other:?}"),
            }
        });
    }

    #[test]
    fn recv_sys_reports_a_closed_sys_lane_and_leaves_data_alone() {
        block_on(async {
            let (sys_tx, mut data_tx, mut inb) = lanes();
            data_tx
                .try_send(Stamped {
                    seq: 1,
                    frame: Transcript::user_final("still here").into(),
                })
                .unwrap();
            drop(sys_tx);

            assert!(
                matches!(inb.recv_sys().now_or_never(), Some(None)),
                "a closed sys lane must resolve immediately, so a racing caller can stop polling",
            );
            match inb.recv().await.unwrap() {
                Received::Data(DataFrame::Transcript(s)) => {
                    assert_eq!(s.text, "still here".into())
                }
                other => panic!("recv_sys must not consume data frames, got {other:?}"),
            }
        });
    }
}
