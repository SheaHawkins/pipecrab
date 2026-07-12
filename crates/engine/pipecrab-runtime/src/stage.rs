//! The asynchronous half of a pipeline stage and its preemptible run loop.
//!
//! A [`Stage`] combines a sans-I/O [`Processor`](pipecrab_core::Processor) with
//! [`Stage::perform`]. The processor owns mutable state and emits effects;
//! `perform` executes those effects through `&self`. This split lets the run
//! loop cancel in-flight I/O without leaving stage state half-updated.
//!
//! [`Stage::run`] connects a stage to [`Inbound`] and [`Outbound`]. Its default
//! drives a leaf stage; composites such as [`Pipeline`](crate::Pipeline)
//! override it to drive children.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::FutureExt;
use futures::pin_mut;
use futures::stream::StreamExt;
use pipecrab_core::{DataFrame, Direction, Disposition, Processor, SystemFrame};

use crate::{Inbound, MaybeSend, MaybeSendSync, Outbound, Received};

/// An error from [`Stage::perform`].
///
/// The run loop sends it upstream as [`SystemFrame::Error`].
#[derive(Debug, Clone)]
pub struct StageError {
    /// Human-readable description.
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

/// The asynchronous, effecting half of a pipeline stage.
///
/// [`Processor`] owns state and emits effects. [`Stage::perform`] executes one
/// effect and emits resulting frames. Keeping fallible I/O out of
/// [`Processor::decide_data`] and [`Processor::decide_system`] makes every state
/// transition synchronous and uninterruptible.
///
/// [`Stage::run`] provides the preemptible leaf loop. Composite stages such as
/// [`Pipeline`](crate::Pipeline) override it.
///
/// # `?Send` is deliberate
///
/// Futures are `Send` on native targets and local on `wasm32`. Use
/// [`offload`](fn@crate::offload) for blocking or CPU-bound work.
///
/// Pipelines erase each stage's effect type when storing heterogeneous stages.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait Stage: Processor + MaybeSendSync
where
    Self::Effect: MaybeSend,
{
    /// Executes one effect and sends resulting frames through `out`.
    ///
    /// This method must not mutate stage state. It may be dropped at an await
    /// point when an interrupt arrives. Use [`offload`] for blocking work.
    ///
    /// [`offload`]: fn@crate::offload
    async fn perform(&self, effect: Self::Effect, out: &Outbound) -> Result<(), StageError>;

    /// Runs until input closes, a stop frame arrives, or an effect fails fatally.
    ///
    /// The default loop prioritizes system frames. An interrupt drops in-flight
    /// effects; other system frames wait until the current effects finish.
    ///
    /// After an interrupt, [`Inbound::flush_data`] discards queued derived data
    /// and reprocesses surviving transport audio first.
    ///
    /// Composite stages override this method; see [`Pipeline`](crate::Pipeline).
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

/// Handles a system frame and returns whether the stage should stop.
async fn handle_system<S: Stage + ?Sized>(
    stage: &mut S,
    dir: Direction,
    frame: SystemFrame,
    out: &Outbound,
) -> bool
where
    S::Effect: MaybeSend,
{
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
) -> Result<(), StageError>
where
    S::Effect: MaybeSend,
{
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
        .send_system(
            Direction::Up,
            SystemFrame::Error {
                message: e.message,
                fatal: e.fatal,
            },
        )
        .await;
}
