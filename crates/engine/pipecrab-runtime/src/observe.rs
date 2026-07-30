//! Pipeline observation: a [`StageObserver`] receives a callback for every
//! frame a stage decides and every effect it performs, across the whole
//! pipeline, without inserting tap stages by hand.
//!
//! Set one with [`PipelineBuilder::observer`](crate::PipelineBuilder::observer).
//! The pipeline then labels each stage with a [`StageId`], appends a terminal
//! [`TAIL_STAGE`] tap so frames leaving the pipeline are observed too, and
//! propagates the observer into nested pipelines automatically.
//!
//! Callbacks fire on the pipeline's own thread, synchronously with the run
//! loop, so an implementation must never block: capture a timestamp, push into
//! a non-blocking channel, and return. Timekeeping deliberately lives in the
//! observer, not here — the runtime stays free of clocks (and of
//! `Instant::now()`, which traps on `wasm32-unknown-unknown`). Durations fall
//! out of the callback pairing instead: `decide_*` runs between `data_in` /
//! `sys_in` and the matching `*_decided`; one `perform` runs between
//! `perform_start` and `perform_end`.
//!
//! When no observer is set, the run loop's only cost is checking an `Option`.

use std::sync::Arc;

use pipecrab_core::{DataFrame, Direction, Disposition, SystemFrame};

use crate::{MaybeSendSync, StageError};

/// Name of the tap stage the builder appends when an observer is set. Its
/// `data_in` / `sys_in` observations are the frames leaving the pipeline.
pub const TAIL_STAGE: &str = "tail";

/// Identifies one stage within an observed pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageId {
    /// Position path: `"2"` for the third top-level stage, `"2.0"` for the
    /// first stage of a pipeline nested at position 2, and so on.
    pub path: Arc<str>,
    /// The stage's [`Processor::name`](pipecrab_core::Processor::name).
    pub name: &'static str,
}

/// How one `perform` call ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PerformOutcome {
    /// Completed without error.
    Ok,
    /// Returned a [`StageError`].
    Err {
        /// The error's message.
        message: Arc<str>,
        /// Whether the error was fatal.
        fatal: bool,
    },
    /// The in-flight future was dropped by a barge-in `Interrupt`.
    Aborted,
}

impl PerformOutcome {
    pub(crate) fn from_error(e: &StageError) -> Self {
        PerformOutcome::Err {
            message: e.message.clone(),
            fatal: e.fatal,
        }
    }
}

/// Receives run-loop callbacks for every observed stage. All methods default
/// to no-ops; implement only what you consume. Must not block (see the module
/// docs).
#[allow(unused_variables)]
pub trait StageObserver: MaybeSendSync {
    /// A (sub)pipeline finished wiring: `stages` lists its children in order.
    /// The top-level pipeline reports first; nested pipelines follow with
    /// dotted paths.
    fn wired(&self, stages: &[StageId]) {}

    /// A data frame reached `stage`; its `decide_data` runs next.
    fn data_in(&self, stage: &StageId, frame: &DataFrame) {}

    /// `stage`'s `decide_data` returned: the frame's `disposition` and how many
    /// effects were queued.
    fn data_decided(&self, stage: &StageId, disposition: Disposition, effects: usize) {}

    /// A system frame reached `stage`; its `decide_system` runs next.
    fn sys_in(&self, stage: &StageId, dir: Direction, frame: &SystemFrame) {}

    /// `stage`'s `decide_system` returned.
    fn sys_decided(&self, stage: &StageId, disposition: Disposition, effects: usize) {}

    /// `stage` is about to `perform` one effect.
    fn perform_start(&self, stage: &StageId) {}

    /// The `perform` started by the matching [`perform_start`](Self::perform_start)
    /// ended with `outcome` — including [`PerformOutcome::Aborted`] when an
    /// `Interrupt` dropped it mid-flight.
    fn perform_end(&self, stage: &StageId, outcome: PerformOutcome) {}
}

/// One stage's binding to the pipeline's observer: the shared observer plus
/// this stage's identity. Carried by [`Inbound`](crate::Inbound) and consulted
/// by the run loop.
#[derive(Clone)]
pub struct ObserverHandle {
    /// The stage these callbacks are about.
    pub stage: StageId,
    /// The pipeline-wide observer.
    pub observer: Arc<dyn StageObserver>,
}

impl ObserverHandle {
    pub(crate) fn data_in(&self, frame: &DataFrame) {
        self.observer.data_in(&self.stage, frame);
    }
    pub(crate) fn data_decided(&self, disposition: Disposition, effects: usize) {
        self.observer
            .data_decided(&self.stage, disposition, effects);
    }
    pub(crate) fn sys_in(&self, dir: Direction, frame: &SystemFrame) {
        self.observer.sys_in(&self.stage, dir, frame);
    }
    pub(crate) fn sys_decided(&self, disposition: Disposition, effects: usize) {
        self.observer.sys_decided(&self.stage, disposition, effects);
    }
    pub(crate) fn perform_start(&self) {
        self.observer.perform_start(&self.stage);
    }
    pub(crate) fn perform_end(&self, outcome: PerformOutcome) {
        self.observer.perform_end(&self.stage, outcome);
    }
}
