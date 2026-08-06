//! [`TelemetryHub`]: the [`StageObserver`] that turns run-loop callbacks into
//! timestamped [`Event`]s on a bounded channel.
//!
//! Every callback is: read the clock, build an owned event, `try_send`. The
//! pipeline thread never blocks or allocates unboundedly on telemetry; if the
//! consumer falls behind and the channel fills, events are dropped, counted,
//! and the loss is reported in-band as [`EventKind::Lost`] once space frees.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};

use pipecrab_core::{DataFrame, Direction, Disposition, SystemFrame};
use pipecrab_runtime::{PerformOutcome, StageId, StageObserver};

use crate::clock::Clock;
use crate::event::{Event, EventKind};

/// Default bound of the export channel, in events. At voice-pipeline rates
/// (tens of frames a second across a handful of stages) this is seconds of
/// headroom for a stalled consumer.
pub const DEFAULT_CAPACITY: usize = 4096;

/// The observer half of telemetry: hand
/// [`observer()`](TelemetryHub::observer) (an `Arc<Self>`) to
/// [`PipelineBuilder::observer`](pipecrab_runtime::PipelineBuilder::observer)
/// and drain the [`Receiver`] returned alongside — natively the
/// [`Session`](crate::Session) worker does the draining.
pub struct TelemetryHub {
    clock: Arc<dyn Clock>,
    tx: SyncSender<Event>,
    dropped: AtomicU64,
}

impl TelemetryHub {
    /// A hub over `clock` with the [`DEFAULT_CAPACITY`] channel bound, plus
    /// the receiving end its events arrive on.
    pub fn new(clock: Arc<dyn Clock>) -> (Arc<Self>, Receiver<Event>) {
        Self::with_capacity(clock, DEFAULT_CAPACITY)
    }

    /// A hub with an explicit channel bound (clamped to at least 1).
    pub fn with_capacity(clock: Arc<dyn Clock>, capacity: usize) -> (Arc<Self>, Receiver<Event>) {
        let (tx, rx) = sync_channel(capacity.max(1));
        (
            Arc::new(Self {
                clock,
                tx,
                dropped: AtomicU64::new(0),
            }),
            rx,
        )
    }

    /// This hub as the [`StageObserver`] to give the pipeline builder.
    pub fn observer(self: &Arc<Self>) -> Arc<dyn StageObserver> {
        self.clone()
    }

    /// Events dropped so far because the channel was full and not yet
    /// reported in-band.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    fn send(&self, stage: Option<&StageId>, kind: EventKind) {
        let at = self.clock.now();
        // Report any earlier loss in-band first, so the consumer knows the
        // stream has a gap at this point.
        let lost = self.dropped.swap(0, Ordering::Relaxed);
        if lost > 0 {
            let report = Event {
                at,
                stage: None,
                kind: EventKind::Lost { count: lost },
            };
            if self.tx.try_send(report).is_err() {
                // Still full: restore the count and drop this event too.
                self.dropped.fetch_add(lost + 1, Ordering::Relaxed);
                return;
            }
        }
        let event = Event {
            at,
            stage: stage.cloned(),
            kind,
        };
        if let Err(TrySendError::Full(_)) = self.tx.try_send(event) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        // A disconnected receiver means telemetry is shutting down; events are
        // discarded silently.
    }
}

impl StageObserver for TelemetryHub {
    fn wired(&self, stages: &[StageId]) {
        self.send(
            None,
            EventKind::Wired {
                stages: stages.to_vec(),
            },
        );
    }

    fn data_in(&self, stage: &StageId, frame: &DataFrame) {
        self.send(Some(stage), EventKind::FrameIn(frame.into()));
    }

    fn data_decided(&self, stage: &StageId, disposition: Disposition, effects: usize) {
        self.send(
            Some(stage),
            EventKind::Decided {
                disposition,
                effects,
            },
        );
    }

    fn sys_in(&self, stage: &StageId, dir: Direction, frame: &SystemFrame) {
        self.send(
            Some(stage),
            EventKind::SysIn {
                dir,
                frame: frame.clone(),
            },
        );
    }

    fn sys_decided(&self, stage: &StageId, disposition: Disposition, effects: usize) {
        self.send(
            Some(stage),
            EventKind::SysDecided {
                disposition,
                effects,
            },
        );
    }

    fn perform_start(&self, stage: &StageId) {
        self.send(Some(stage), EventKind::PerformStart);
    }

    fn perform_end(&self, stage: &StageId, outcome: PerformOutcome) {
        self.send(Some(stage), EventKind::PerformEnd { outcome });
    }
}
