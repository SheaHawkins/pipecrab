//! [`Pipeline`]: wires a sequence of [`Stage`]s into one runnable, preemptible
//! future.
//!
//! # Topology (v1)
//!
//! Both lanes form a linear downstream chain: an external `data_in`/`sys_in`
//! feeds stage 0, each stage's output feeds the next, and the tail's output is
//! the external `data_out`/`sys_out`. Adjacent stages are joined by a `data`
//! channel, and the `sys` lane is threaded through every stage the same way, so
//! a control frame visits each stage in turn. Closing the inputs therefore
//! cascades shutdown cleanly from head to tail.
//!
//! Upstream routing of `Up`-travelling system frames *through* the stages
//! (e.g. an `Error` flowing back toward the source) is not yet wired: for now a
//! [`Stage::perform`] error is surfaced on the tail's `sys_out`, tagged
//! [`Direction::Up`]. That is a deliberate v1 limitation.
//!
//! # Driving it
//!
//! [`Pipeline::run`] is a single future that drives every stage's run loop
//! cooperatively via [`FuturesUnordered`]; there is no spawning and no executor
//! trait. The caller drives it — `block_on` natively, `spawn_local` in the
//! browser.

use futures::channel::mpsc;
use futures::future::FutureExt;
use futures::pin_mut;
use futures::stream::{FuturesUnordered, StreamExt};
use pipecrab_core::{DataFrame, Direction, Disposition, SystemFrame};

use crate::{Inbound, Outbound, Received, Stage, StageError};

/// Default capacity for each inter-stage lane channel.
const DEFAULT_CAPACITY: usize = 16;

/// The external endpoints of a built [`Pipeline`]: feed the head, read the tail.
///
/// Returned alongside the pipeline from [`PipelineBuilder::build`]. The caller
/// holds these while [`Pipeline::run`] consumes the pipeline itself. Dropping
/// both input senders closes the head's inbound lanes, which cascades shutdown
/// downstream and lets `run` return.
pub struct PipelineEnds {
    /// Sends data frames into the head stage's data lane.
    pub data_in: mpsc::Sender<DataFrame>,
    /// Sends system frames into the head stage's system lane.
    pub sys_in: mpsc::Sender<(Direction, SystemFrame)>,
    /// Receives data frames emitted past the tail stage.
    pub data_out: mpsc::Receiver<DataFrame>,
    /// Receives system frames emitted past the tail stage (including
    /// [`Direction::Up`]-tagged errors from any stage's `perform`).
    pub sys_out: mpsc::Receiver<(Direction, SystemFrame)>,
}

/// Builds a [`Pipeline`] from an ordered list of stages sharing one `Effect`.
pub struct PipelineBuilder<E> {
    stages: Vec<Box<dyn Stage<Effect = E>>>,
    capacity: usize,
}

impl<E: 'static> PipelineBuilder<E> {
    /// A new, empty builder with the default lane capacity.
    pub fn new() -> Self {
        Self { stages: Vec::new(), capacity: DEFAULT_CAPACITY }
    }

    /// Override the per-lane channel capacity (clamped to at least 1).
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(1);
        self
    }

    /// Append a stage, boxing it. Stages run in the order added, head first.
    pub fn stage(mut self, stage: impl Stage<Effect = E> + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Append an already-boxed stage.
    pub fn boxed(mut self, stage: Box<dyn Stage<Effect = E>>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Wire the stages into a runnable [`Pipeline`] and its external
    /// [`PipelineEnds`].
    ///
    /// # Panics
    ///
    /// Panics if no stages were added — a pipeline needs at least one stage.
    pub fn build(self) -> (PipelineEnds, Pipeline<E>) {
        assert!(!self.stages.is_empty(), "a pipeline needs at least one stage");
        let cap = self.capacity;

        // `prev_*_rx` starts as the external input's receiver and, after the
        // loop, ends up as the tail's output receiver — the external output.
        let (data_in, mut prev_data_rx) = mpsc::channel::<DataFrame>(cap);
        let (sys_in, mut prev_sys_rx) = mpsc::channel::<(Direction, SystemFrame)>(cap);

        let mut cells = Vec::with_capacity(self.stages.len());
        for stage in self.stages {
            let (data_tx, next_data_rx) = mpsc::channel::<DataFrame>(cap);
            let (sys_tx, next_sys_rx) = mpsc::channel::<(Direction, SystemFrame)>(cap);
            cells.push(StageCell {
                stage,
                inbound: Inbound { sys: prev_sys_rx, data: prev_data_rx },
                out: Outbound { data: data_tx, sys: sys_tx },
            });
            prev_data_rx = next_data_rx;
            prev_sys_rx = next_sys_rx;
        }

        let ends = PipelineEnds { data_in, sys_in, data_out: prev_data_rx, sys_out: prev_sys_rx };
        (ends, Pipeline { cells })
    }
}

impl<E: 'static> Default for PipelineBuilder<E> {
    fn default() -> Self {
        Self::new()
    }
}

/// A wired pipeline: a sequence of stages plus their channels, ready to [`run`].
///
/// [`run`]: Pipeline::run
pub struct Pipeline<E> {
    cells: Vec<StageCell<E>>,
}

struct StageCell<E> {
    stage: Box<dyn Stage<Effect = E>>,
    inbound: Inbound,
    out: Outbound,
}

impl<E: 'static> Pipeline<E> {
    /// Drive every stage's run loop to completion as one cooperative future.
    ///
    /// Returns once all stages have shut down (their inbound lanes closed). No
    /// spawning happens here — the caller drives the returned future.
    pub async fn run(self) {
        let mut tasks: FuturesUnordered<_> =
            self.cells.into_iter().map(|c| run_stage(c.stage, c.inbound, c.out)).collect();
        while tasks.next().await.is_some() {}
    }
}

/// The per-stage preemptible run loop.
///
/// System frames are drained before data (via [`Inbound::recv`]). While a data
/// frame's effects run in `perform`, the system lane is raced against them: an
/// `Interrupt` drops the in-flight `perform` immediately; any other system
/// frame is stashed and handled once `perform` is dropped (we cannot call the
/// `&mut self` `decide_system` while `perform` borrows `&self`). The loop ends
/// when both inbound lanes close, on a `Stop`, or on a fatal error.
async fn run_stage<E>(mut stage: Box<dyn Stage<Effect = E>>, mut inbound: Inbound, out: Outbound) {
    while let Some(received) = inbound.recv().await {
        match received {
            Received::Sys(dir, frame) => {
                if handle_system(stage.as_mut(), dir, frame, &out).await {
                    break;
                }
            }
            Received::Data(frame) => {
                let decision = stage.decide_data(&frame);
                if decision.disposition == Disposition::Forward {
                    let _ = out.send_data(frame).await;
                }
                if decision.effects.is_empty() {
                    continue;
                }

                let mut stashed: Vec<(Direction, SystemFrame)> = Vec::new();
                let mut interrupt: Option<(Direction, SystemFrame)> = None;
                let mut should_stop = false;
                {
                    // `perform` borrows `&stage` for its whole lifetime, so no
                    // `&mut stage` (i.e. no `decide_system`) is possible until it
                    // is dropped at the end of this block.
                    let perform = run_effects(stage.as_ref(), decision.effects, &out).fuse();
                    pin_mut!(perform);
                    loop {
                        futures::select_biased! {
                            maybe = inbound.sys.next() => {
                                // `None` => sys lane closed; fall through and keep
                                // awaiting `perform`.
                                if let Some((d, f)) = maybe {
                                    if matches!(f, SystemFrame::Interrupt) {
                                        interrupt = Some((d, f));
                                        break; // drops `perform`: barge-in
                                    }
                                    stashed.push((d, f)); // defer; keep performing
                                }
                            },
                            res = perform => {
                                if let Err(e) = res {
                                    let fatal = e.fatal;
                                    emit_error(&out, e).await;
                                    should_stop |= fatal;
                                }
                                break;
                            },
                            complete => break,
                        }
                    }
                }

                // `perform` is dropped; `&mut stage` is free again.
                for (d, f) in stashed.drain(..) {
                    should_stop |= handle_system(stage.as_mut(), d, f, &out).await;
                }
                if let Some((d, f)) = interrupt {
                    should_stop |= handle_system(stage.as_mut(), d, f, &out).await;
                }
                if should_stop {
                    break;
                }
            }
        }
    }
}

/// Run a system frame through the stage: `decide_system`, forward on `Forward`,
/// then perform its effects. Returns `true` if the stage should stop (the frame
/// was a `Stop`, or an effect failed fatally).
async fn handle_system<E>(
    stage: &mut dyn Stage<Effect = E>,
    dir: Direction,
    frame: SystemFrame,
    out: &Outbound,
) -> bool {
    let mut should_stop = matches!(frame, SystemFrame::Stop);
    let decision = stage.decide_system(dir, &frame);
    if decision.disposition == Disposition::Forward {
        let _ = out.send_system(dir, frame).await;
    }
    for effect in decision.effects {
        if let Err(e) = stage.perform(effect, out).await {
            let fatal = e.fatal;
            emit_error(out, e).await;
            should_stop |= fatal;
        }
    }
    should_stop
}

/// Perform a stage's effects in order, short-circuiting on the first error.
async fn run_effects<E>(
    stage: &dyn Stage<Effect = E>,
    effects: Vec<E>,
    out: &Outbound,
) -> Result<(), StageError> {
    for effect in effects {
        stage.perform(effect, out).await?;
    }
    Ok(())
}

/// Surface a `perform` failure as an `Error` system frame. v1 sends it on the
/// downstream `sys` lane tagged [`Direction::Up`]; true upstream routing is a
/// follow-up.
async fn emit_error(out: &Outbound, e: StageError) {
    let _ = out
        .send_system(Direction::Up, SystemFrame::Error { message: e.message, fatal: e.fatal })
        .await;
}
