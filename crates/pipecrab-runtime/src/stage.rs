//! The [`Stage`] trait: the async, effecting half of a pipeline stage, and the
//! preemptible run loop ([`Stage::run`]) that drives one.
//!
//! A stage is a [`Processor`] — synchronous, state-owning `decide_*` — plus an
//! async [`Stage::perform`] that interprets the effects `decide_*` emitted and
//! does the actual I/O. The split is the core invariant: `decide_*` takes
//! `&mut self` and is the *only* place state changes; `perform` takes `&self`
//! and must never mutate state, so the run loop can drop an in-flight `perform`
//! future on an interrupt without leaving torn state behind.
//!
//! [`Stage::run`] ties a stage to an [`Inbound`] and an [`Outbound`] and drives
//! it. Its default body is the leaf run loop; a composite stage (a
//! [`Pipeline`](crate::Pipeline)) overrides it to drive its children — which is
//! why a pipeline is itself a `Stage` and can nest.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::FutureExt;
use futures::pin_mut;
use futures::stream::StreamExt;
use pipecrab_core::{DataFrame, Direction, Disposition, Processor, SystemFrame};

use crate::{Inbound, Outbound, Received};

/// Why a [`Stage::perform`] call failed.
///
/// `perform` is the fallible, I/O-doing half of a stage. The run loop surfaces
/// a returned error as a `SystemFrame::Error` travelling upstream; `fatal`
/// decides whether the pipeline should tear down rather than carry on.
///
/// Mirrors the shape of `SystemFrame::Error` (a message plus a `fatal` flag) so
/// the conversion at the run-loop boundary is direct.
#[derive(Debug, Clone)]
pub struct StageError {
    /// Human-readable description of what went wrong.
    pub message: Arc<str>,
    /// Whether the failure is unrecoverable and the pipeline should shut down.
    pub fatal: bool,
}

impl StageError {
    /// A recoverable error: the pipeline may keep running.
    pub fn new(message: impl Into<Arc<str>>) -> Self {
        Self { message: message.into(), fatal: false }
    }

    /// An unrecoverable error: the pipeline should shut down.
    pub fn fatal(message: impl Into<Arc<str>>) -> Self {
        Self { message: message.into(), fatal: true }
    }
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.fatal { "fatal stage error" } else { "stage error" };
        write!(f, "{kind}: {}", self.message)
    }
}

impl std::error::Error for StageError {}

impl From<String> for StageError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for StageError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

/// The async, effecting half of a pipeline stage.
///
/// `Stage` extends [`Processor`]: `decide_data` / `decide_system` (synchronous,
/// `&mut self`) own all state mutation and emit [`Effect`](Processor::Effect)
/// values; [`perform`](Stage::perform) interprets one effect, does its I/O, and
/// pushes any resulting frames through `out`.
///
/// [`run`](Stage::run) drives the stage given an [`Inbound`] and an
/// [`Outbound`]. Its default is the preemptible leaf loop; a composite stage
/// overrides it (see [`Pipeline`](crate::Pipeline)), which is what lets a
/// pipeline be a `Stage` and nest inside another.
///
/// # `?Send` is deliberate
///
/// pipecrab commits to a single-threaded execution model, so the returned
/// futures are **not** required to be `Send`. One `Stage` definition then runs
/// unchanged both on a tokio current-thread runtime and in the browser
/// (`wasm32`), where `Send` bounds are impossible to satisfy. CPU-bound or
/// blocking work must not run inline on the orchestrator thread — push it
/// off-thread with the `offload` helper and `await` the result, so an interrupt
/// can still preempt `perform` promptly.
///
/// The trait is dyn-compatible (via `async_trait`), so a pipeline can hold its
/// stages as `Box<dyn Stage<Effect = _>>`.
#[async_trait(?Send)]
pub trait Stage: Processor {
    /// Interpret one effect emitted by `decide_*` and carry out its I/O, sending
    /// any resulting frames through `out`.
    ///
    /// Takes `&self`: `perform` must not mutate stage state. The run loop races
    /// this future against the system lane, so a barge-in `Interrupt` can drop
    /// it mid-flight; because only `decide_*` ever mutated state, dropping the
    /// future leaves the stage intact. Barge-in is only as responsive as
    /// `perform` yields, so never block the thread inline — offload heavy work
    /// and `await` it.
    async fn perform(&self, effect: Self::Effect, out: &Outbound) -> Result<(), StageError>;

    /// Drive this stage to completion: consume frames from `inbound`, emit
    /// through `out`, return once `inbound` closes (or on `Stop` / a fatal
    /// error).
    ///
    /// The default is the preemptible run loop. System frames are drained
    /// before data (via [`Inbound::recv`]). While a data frame's effects run in
    /// `perform`, the system lane is raced against them: an `Interrupt` drops
    /// the in-flight `perform` immediately; any other system frame is *stashed*
    /// and handled once `perform` is dropped — we cannot call the `&mut self`
    /// `decide_system` while `perform` borrows `&self`, so the stash defers it
    /// until that borrow ends.
    ///
    /// After an `Interrupt` is handled, the queued data backlog is flushed via
    /// [`Inbound::flush_data`]: droppable frames are discarded, but
    /// transport-audio survivors are kept and re-processed ahead of the next
    /// inbound read, so a barge-in utterance is not clipped.
    ///
    /// A composite stage overrides this; the default body is never invoked for
    /// one (see [`Pipeline`](crate::Pipeline)).
    async fn run(self: Box<Self>, inbound: Inbound, out: Outbound) {
        let mut stage = self;
        let mut inbound = inbound;
        // Survivors of an interrupt flush, re-processed ahead of the next read.
        let mut pending: VecDeque<DataFrame> = VecDeque::new();
        loop {
            let received = match pending.pop_front() {
                Some(frame) => Received::Data(frame),
                None => match inbound.recv().await {
                    Some(received) => received,
                    None => break,
                },
            };
            match received {
                Received::Sys(dir, frame) => {
                    let interrupted = matches!(frame, SystemFrame::Interrupt);
                    let stop = handle_system(&mut *stage, dir, frame, &out).await;
                    if interrupted {
                        // Barge-in: discard the queued data backlog, but keep
                        // transport-audio survivors and re-process them so the
                        // new utterance is not clipped.
                        pending.extend(inbound.flush_data());
                    }
                    if stop {
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
                        // `perform` borrows `&*stage` for its whole lifetime, so
                        // no `&mut *stage` (i.e. no `decide_system`) is possible
                        // until it is dropped at the end of this block.
                        let perform = run_effects(&*stage, decision.effects, &out).fuse();
                        pin_mut!(perform);
                        loop {
                            futures::select_biased! {
                                maybe = inbound.sys.next() => {
                                    // `None` => sys lane closed; keep performing.
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

                    // `perform` is dropped; `&mut *stage` is free again.
                    for (d, f) in stashed.drain(..) {
                        should_stop |= handle_system(&mut *stage, d, f, &out).await;
                    }
                    if let Some((d, f)) = interrupt {
                        should_stop |= handle_system(&mut *stage, d, f, &out).await;
                        // Same barge-in flush as the outer Sys branch.
                        pending.extend(inbound.flush_data());
                    }
                    if should_stop {
                        break;
                    }
                }
            }
        }
    }
}

/// Run a system frame through the stage: `decide_system`, forward on `Forward`,
/// then perform its effects. Returns `true` if the stage should stop (the frame
/// was a `Stop`, or an effect failed fatally).
async fn handle_system<S: Stage + ?Sized>(
    stage: &mut S,
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
async fn run_effects<S: Stage + ?Sized>(
    stage: &S,
    effects: Vec<S::Effect>,
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
