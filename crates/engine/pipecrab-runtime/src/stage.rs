//! The [`Stage`] trait: the async, effecting half of a pipeline stage, and the
//! preemptible run loop ([`Stage::run`]) that drives one.
//!
//! A stage is a [`Processor`](pipecrab_core::Processor) — synchronous,
//! state-owning `decide_*` — plus an
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
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::future::FutureExt;
use futures::pin_mut;
use futures::stream::StreamExt;
use pipecrab_core::{DataFrame, Direction, Disposition, Processor, SystemFrame};

use crate::inbound::Stamped;
use crate::observe::{ObserverHandle, PerformOutcome};
use crate::{Inbound, MaybeSend, MaybeSendSync, Outbound, Received};

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
        Self {
            message: message.into(),
            fatal: false,
        }
    }

    /// An unrecoverable error: the pipeline should shut down.
    pub fn fatal(message: impl Into<Arc<str>>) -> Self {
        Self {
            message: message.into(),
            fatal: true,
        }
    }
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.fatal {
            "fatal stage error"
        } else {
            "stage error"
        };
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
/// off-thread with [`offload`](fn@crate::offload) and `await` the result, so an
/// interrupt can still preempt `perform` promptly.
///
/// The trait is dyn-compatible (via `async_trait`). A pipeline erases the
/// associated effect type at insertion and stores only an object-safe runner,
/// allowing stages with different effect types to compose.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait Stage: Processor + MaybeSendSync
where
    Self::Effect: MaybeSend,
{
    /// Interpret one effect emitted by `decide_*` and carry out its I/O, sending
    /// any resulting frames through `out`.
    ///
    /// Takes `&self`: `perform` must not mutate stage state. The run loop races
    /// this future against the system lane, so a barge-in `Interrupt` can drop
    /// it mid-flight; because only `decide_*` ever mutated state, dropping the
    /// future leaves the stage intact. Barge-in is only as responsive as
    /// `perform` yields, so never block the thread inline — [`offload`] heavy
    /// work and `await` it.
    ///
    /// [`offload`]: fn@crate::offload
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
    /// [`Inbound::flush_data`]: droppable frames queued before the `Interrupt`
    /// are discarded; survivors and frames queued after it are kept and
    /// re-processed ahead of the next inbound read, so a barge-in utterance is
    /// not clipped. The replay itself yields to any system frame already
    /// queued — the sys lane keeps its priority — and a later interrupt
    /// re-judges the held keepers by their stamps.
    ///
    /// A composite stage overrides this; the default body is never invoked for
    /// one (see [`Pipeline`](crate::Pipeline)).
    async fn run(self: Box<Self>, inbound: Inbound, out: Outbound) {
        let mut stage = self;
        let mut inbound = inbound;
        let obs = inbound.observer.clone();
        let obs = obs.as_ref();
        // Keepers of an interrupt flush, re-processed ahead of the next read.
        // Stamped so a later interrupt's flush can re-judge them by seq.
        let mut pending: VecDeque<Stamped<DataFrame>> = VecDeque::new();
        loop {
            let received = if pending.is_empty() {
                match inbound.recv().await {
                    Some(received) => received,
                    None => break,
                }
            } else {
                // Replaying keepers must not starve the sys lane: a system
                // frame already queued keeps the priority recv() would give it.
                match inbound.try_recv_sys() {
                    Some((dir, frame)) => Received::Sys(dir, frame),
                    None => Received::Data(pending.pop_front().expect("non-empty").frame),
                }
            };
            match received {
                Received::Sys(dir, frame) => {
                    let interrupted = matches!(frame, SystemFrame::Interrupt);
                    let stop = handle_system(&mut *stage, dir, frame, &out, obs).await;
                    if interrupted {
                        // Barge-in: discard the stale queued data backlog, but
                        // keep survivors and anything queued after the
                        // Interrupt, re-processing them so the new utterance is
                        // not clipped. Held keepers are re-judged the same way.
                        let floor = inbound.flush_floor;
                        pending.retain(|s| s.seq >= floor || s.frame.survives_flush());
                        pending.extend(inbound.flush_data_stamped());
                    }
                    if stop {
                        break;
                    }
                }
                Received::Data(frame) => {
                    if let Some(h) = obs {
                        h.data_in(&frame);
                    }
                    let decision = stage.decide_data(&frame);
                    if let Some(h) = obs {
                        h.data_decided(decision.disposition, decision.effects.len());
                    }
                    if decision.disposition == Disposition::Forward {
                        let _ = out.send_data(frame).await;
                    }
                    if decision.effects.is_empty() {
                        continue;
                    }

                    let mut stashed: Vec<(Direction, SystemFrame)> = Vec::new();
                    let mut interrupt: Option<(u64, Direction, SystemFrame)> = None;
                    let mut should_stop = false;
                    // True while a `perform_start` has been reported without its
                    // matching `perform_end` — i.e. an effect is in flight. Lets
                    // the abort path below close the observer's open pair.
                    let perform_open = AtomicBool::new(false);
                    {
                        // `perform` borrows `&*stage` for its whole lifetime, so
                        // no `&mut *stage` (i.e. no `decide_system`) is possible
                        // until it is dropped at the end of this block.
                        let perform =
                            run_effects(&*stage, decision.effects, &out, obs, &perform_open).fuse();
                        pin_mut!(perform);
                        loop {
                            futures::select_biased! {
                                maybe = inbound.sys.next() => {
                                    // `None` => sys lane closed; keep performing.
                                    if let Some(Stamped { seq, frame: (d, f) }) = maybe {
                                        if matches!(f, SystemFrame::Interrupt) {
                                            interrupt = Some((seq, d, f));
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
                    if perform_open.load(Ordering::Relaxed) {
                        // The dropped future reported `perform_start` for the
                        // effect in flight; close that pair as aborted.
                        if let Some(h) = obs {
                            h.perform_end(PerformOutcome::Aborted);
                        }
                    }
                    for (d, f) in stashed.drain(..) {
                        should_stop |= handle_system(&mut *stage, d, f, &out, obs).await;
                    }
                    if let Some((seq, d, f)) = interrupt {
                        // This path took the frame straight off the sys lane,
                        // so record the floor `recv` would have.
                        inbound.flush_floor = seq;
                        should_stop |= handle_system(&mut *stage, d, f, &out, obs).await;
                        // Same barge-in flush as the outer Sys branch.
                        pending.retain(|s| s.seq >= seq || s.frame.survives_flush());
                        pending.extend(inbound.flush_data_stamped());
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
    obs: Option<&ObserverHandle>,
) -> bool
where
    S::Effect: MaybeSend,
{
    let mut should_stop = matches!(frame, SystemFrame::Stop);
    if let Some(h) = obs {
        h.sys_in(dir, &frame);
    }
    let decision = stage.decide_system(dir, &frame);
    if let Some(h) = obs {
        h.sys_decided(decision.disposition, decision.effects.len());
    }
    if decision.disposition == Disposition::Forward {
        let _ = out.send_system(dir, frame).await;
    }
    for effect in decision.effects {
        if let Some(h) = obs {
            h.perform_start();
        }
        let res = stage.perform(effect, out).await;
        if let Some(h) = obs {
            h.perform_end(match &res {
                Ok(()) => PerformOutcome::Ok,
                Err(e) => PerformOutcome::from_error(e),
            });
        }
        if let Err(e) = res {
            let fatal = e.fatal;
            emit_error(out, e).await;
            should_stop |= fatal;
        }
    }
    should_stop
}

/// Perform a stage's effects in order, short-circuiting on the first error.
///
/// `open` mirrors whether a `perform_start` has been reported without its
/// `perform_end`, so the run loop can close the pair as
/// [`PerformOutcome::Aborted`] if this future is dropped mid-effect.
async fn run_effects<S: Stage + ?Sized>(
    stage: &S,
    effects: Vec<S::Effect>,
    out: &Outbound,
    obs: Option<&ObserverHandle>,
    open: &AtomicBool,
) -> Result<(), StageError>
where
    S::Effect: MaybeSend,
{
    for effect in effects {
        if let Some(h) = obs {
            h.perform_start();
            open.store(true, Ordering::Relaxed);
        }
        let res = stage.perform(effect, out).await;
        if let Some(h) = obs {
            open.store(false, Ordering::Relaxed);
            h.perform_end(match &res {
                Ok(()) => PerformOutcome::Ok,
                Err(e) => PerformOutcome::from_error(e),
            });
        }
        res?;
    }
    Ok(())
}

/// Surface a `perform` failure as an `Error` system frame. v1 sends it on the
/// downstream `sys` lane tagged [`Direction::Up`]; true upstream routing is a
/// follow-up.
async fn emit_error(out: &Outbound, e: StageError) {
    let _ = out
        .send_system(
            Direction::Up,
            SystemFrame::Error {
                message: e.message,
                fatal: e.fatal,
            },
        )
        .await;
}
