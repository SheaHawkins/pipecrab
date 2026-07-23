//! [`DispatchIngress`]: the *active* pass-through stage that lets external
//! dispatch events enter an otherwise idle pipeline.
//!
//! # Active, not frame-driven
//!
//! An ordinary leaf stage only wakes when a frame arrives on its inbound lanes.
//! Dispatch events arrive from *outside* the pipeline, on no lane, and often
//! while nothing else is flowing (a task finishing minutes after the user spoke).
//! So ingress overrides [`Stage::run`] and concurrently polls three things — the
//! system lane, the external [`DispatchSource`], and the inbound data lane — as
//! one future. This keeps the pipeline's single driver the *only* future the
//! application drives: there is no separate listener task to spawn.
//!
//! # Raw event, then projection
//!
//! For every external event ingress emits the raw native frame first —
//! `DataFrame::Dispatch(DispatchFrame::Event(..))`, authoritative and available
//! to any downstream observer — and then a derived
//! `DataFrame::Model(ModelFrame::Input(..))` projection that shapes conversation
//! behavior. The two are distinct on purpose: the raw event is the record; the
//! projection is only how the model should react to it.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::FutureExt;
use futures::pin_mut;
use pipecrab_core::{
    DataFrame, Decision, Direction, DispatchEvent, DispatchFrame, ModelFrame, ModelInput,
    ModelMessage, Processor, SystemFrame,
};
use pipecrab_runtime::{Inbound, Outbound, Received, Stage, StageError};

use crate::error::DispatchError;
use crate::transport::DispatchSource;

/// An active pass-through stage: ordinary frames flow through untouched while,
/// concurrently, events from an external [`DispatchSource`] are injected into the
/// data lane.
///
/// # Lifecycle
///
/// * `Start` is forwarded normally.
/// * `Interrupt` is forwarded and does *not* stop dispatch listening — a
///   barge-in never deafens the pipeline to external tasks.
/// * `Stop` (and closure of both inbound lanes) cancels the source and
///   terminates ingress.
/// * Source closure (`Ok(None)`) stops source polling but leaves the ordinary
///   pipeline input flowing.
/// * A source error becomes a recoverable or fatal
///   [`StageError`](pipecrab_runtime::StageError) per its
///   [`DispatchError`] classification; a fatal one terminates ingress.
pub struct DispatchIngress<S> {
    // A `Stage` must be `Send + Sync`, but a `DispatchSource` is only `Send`.
    // `Mutex<S>` is `Sync` when `S: Send`, so it makes the stage `Sync` without
    // demanding `Sync` of the source. Never locked — `run` takes ownership via
    // `into_inner`.
    source: Mutex<S>,
}

impl<S> DispatchIngress<S> {
    /// Wrap a [`DispatchSource`] as the pipeline's ingress stage.
    pub fn new(source: S) -> Self {
        Self {
            source: Mutex::new(source),
        }
    }
}

// Like a `Pipeline`, ingress lives entirely in its overridden `run`; the leaf
// `decide_*` / `perform` are never reached and panic as misuse tripwires.
impl<S: DispatchSource> Processor for DispatchIngress<S> {
    type Effect = ();

    fn decide_data(&mut self, _frame: &DataFrame) -> Decision<()> {
        unreachable!("DispatchIngress is driven by Stage::run, not decide_data")
    }

    fn decide_system(&mut self, _dir: Direction, _frame: &SystemFrame) -> Decision<()> {
        unreachable!("DispatchIngress is driven by Stage::run, not decide_system")
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<S: DispatchSource> Stage for DispatchIngress<S> {
    async fn perform(&self, _effect: (), _out: &Outbound) -> Result<(), StageError> {
        unreachable!("DispatchIngress is driven by Stage::run, not perform")
    }

    async fn run(self: Box<Self>, inbound: Inbound, out: Outbound) {
        let DispatchIngress { source } = *self;
        // Never poisoned: the lock is never taken, only consumed here.
        let mut source = source.into_inner().unwrap_or_else(|e| e.into_inner());
        let mut inbound = inbound;
        // Once the source closes we stop polling it but keep serving the lanes.
        let mut source_open = true;

        loop {
            let received = if source_open {
                // Fresh futures each iteration; both are cancellation-safe (an
                // mpsc `recv` and the source's documented contract), so the
                // loser being dropped loses nothing.
                let recv = inbound.recv().fuse();
                let event = source.next_event().fuse();
                pin_mut!(recv, event);
                // Bias toward the lanes so a `Stop` is not starved by a busy
                // source.
                futures::select_biased! {
                    received = recv => received,
                    step = event => {
                        match step {
                            Ok(Some(event)) => {
                                inject(&event, &out).await;
                                continue;
                            }
                            // Source closed: stop polling it, keep the pipeline
                            // input flowing.
                            Ok(None) => {
                                source_open = false;
                                continue;
                            }
                            Err(error) => {
                                let fatal = error.fatal;
                                emit_error(&out, error).await;
                                if fatal {
                                    break;
                                }
                                continue;
                            }
                        }
                    }
                }
            } else {
                inbound.recv().await
            };

            if forward_inbound(received, &out).await {
                break;
            }
        }

        // Idempotent: safe on every exit path (Stop, lane closure, fatal error).
        source.cancel();
    }
}

/// Forward one received inbound frame. Returns `true` when ingress should stop:
/// a `Stop` system frame, or both lanes closed (`None`).
async fn forward_inbound(received: Option<Received>, out: &Outbound) -> bool {
    match received {
        Some(Received::Sys(dir, frame)) => {
            let stop = matches!(frame, SystemFrame::Stop);
            // `Start` and `Interrupt` forward and keep ingress listening.
            let _ = out.send_system(dir, frame).await;
            stop
        }
        Some(Received::Data(frame)) => {
            let _ = out.send_data(frame).await;
            false
        }
        None => true,
    }
}

/// Emit the raw event, then its model projection. Backpressure applies to both
/// downstream sends.
async fn inject(event: &DispatchEvent, out: &Outbound) {
    // Raw native event first: the authoritative record for downstream observers.
    let _ = out
        .send_data(DataFrame::Dispatch(DispatchFrame::Event(event.clone())))
        .await;
    // Then the derived projection that shapes conversation behavior.
    let _ = out
        .send_data(DataFrame::Model(ModelFrame::Input(project(event))))
        .await;
}

/// Project an external event onto the model input it should produce.
///
/// Only `Accepted` becomes a tool result (it answers the `dispatch_task` call);
/// everything else becomes an `Event`. `Progress` is context-only — it becomes
/// visible on the model's next turn without interrupting the user — as is the
/// `Accepted` acknowledgement; the rest warrant a reply.
fn project(event: &DispatchEvent) -> ModelInput {
    match event {
        // The accepted result carries the assigned `task_id` so a later user
        // answer can produce `update_task(task_id = ..)` without forcing an
        // immediate assistant response — hence context, not respond.
        DispatchEvent::Accepted {
            tool_call_id,
            task_id,
        } => {
            let content = serde_json::json!({ "task_id": task_id.as_ref() }).to_string();
            ModelInput::Context(ModelMessage::ToolResult {
                tool_call_id: tool_call_id.clone(),
                name: Arc::from("dispatch_task"),
                content: Arc::from(content),
            })
        }
        DispatchEvent::Rejected { message, .. } => respond_event("rejected", message),
        DispatchEvent::Progress { message, .. } => context_event("progress", message),
        DispatchEvent::Question { message, .. } => respond_event("question", message),
        DispatchEvent::Completion { message, .. } => respond_event("completion", message),
        DispatchEvent::Failure { message, .. } => respond_event("failure", message),
    }
}

/// A dispatch event the model should react to on its next turn without a reply.
fn context_event(kind: &str, message: &Arc<str>) -> ModelInput {
    ModelInput::Context(dispatch_event(kind, message))
}

/// A dispatch event that warrants an immediate reply.
fn respond_event(kind: &str, message: &Arc<str>) -> ModelInput {
    ModelInput::Respond(dispatch_event(kind, message))
}

fn dispatch_event(kind: &str, message: &Arc<str>) -> ModelMessage {
    ModelMessage::Event {
        source: Arc::from("dispatch"),
        kind: Arc::from(kind),
        content: message.clone(),
    }
}

/// Surface a source error as an `Error` system frame, tagged `Up` like the run
/// loop's own error path.
async fn emit_error(out: &Outbound, error: DispatchError) {
    let _ = out
        .send_system(
            Direction::Up,
            SystemFrame::Error {
                message: error.message,
                fatal: error.fatal,
            },
        )
        .await;
}
