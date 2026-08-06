//! [`DispatchEgress`]: turns model tool calls into [`DispatchCommand`]s,
//! publishes them through a [`DispatchSink`], and echoes them downstream as
//! native dispatch frames.
//!
//! Egress is pure mechanism — a per-tool-call translator. It holds no state: no
//! task map, no `task_id`s, no generation bookkeeping. Durable task state lives
//! in the backend behind the transport.

use std::sync::Arc;

use async_trait::async_trait;
use pipecrab_core::{
    DataFrame, Decision, DispatchCommand, DispatchFrame, ModelFrame, Processor, ToolCall,
};
use pipecrab_runtime::{Outbound, Stage, StageError};
use serde::Deserialize;

use crate::transport::DispatchSink;

/// What [`DispatchEgress::perform`] should do with a matched tool call: publish a
/// translated command, or reject a malformed one as a recoverable error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Publish {
    /// Send this command to the sink, then emit the originating tool call and
    /// the native `Dispatch` command downstream.
    Command {
        /// The translated command bound for the sink.
        command: DispatchCommand,
        /// The originating tool call, re-emitted once the send succeeds.
        call: ToolCall,
    },
    /// A dispatch tool call whose arguments would not parse. Surfaces as a
    /// recoverable [`StageError`]; no command is sent.
    Reject(Arc<str>),
}

/// Translates `dispatch_task` / `update_task` tool calls into
/// [`DispatchCommand`]s, sends them through the [`DispatchSink`], and emits
/// them downstream as native dispatch frames behind the re-emitted
/// [`ModelFrame::ToolCall`]. Unknown tool calls pass through untouched.
///
/// A matched call is *consumed* in `decide_data` and re-emitted only after the
/// sink send succeeds: `ToolCall` frames survive an interrupt flush, so a
/// forwarded-up-front call would outlive a barge-in that dropped the send —
/// a durable claim of a dispatch that never happened. The command itself is
/// still lost when a barge-in drops the send mid-flight; at-least-once
/// delivery needs a transport ack/retry the [`DispatchSink`] trait does not
/// have.
pub struct DispatchEgress<K> {
    sink: K,
}

impl<K> DispatchEgress<K> {
    /// Wrap a [`DispatchSink`] as the pipeline's egress stage.
    pub fn new(sink: K) -> Self {
        Self { sink }
    }
}

impl<K: DispatchSink> Processor for DispatchEgress<K> {
    type Effect = Publish;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<Publish> {
        match frame {
            DataFrame::Model(ModelFrame::ToolCall(call)) => translate(call),
            _ => Decision::forward(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<K: DispatchSink> Stage for DispatchEgress<K> {
    async fn perform(&self, effect: Publish, out: &Outbound) -> Result<(), StageError> {
        match effect {
            Publish::Command { command, call } => {
                // Publish to the transport first; only then claim it
                // downstream. A barge-in dropping this future mid-send thus
                // emits nothing, instead of a surviving ToolCall that claims a
                // dispatch that never happened.
                self.sink.send_command(command.clone()).await?;
                let _ = out
                    .send_data(DataFrame::Model(ModelFrame::ToolCall(call)))
                    .await;
                let _ = out
                    .send_data(DataFrame::Dispatch(DispatchFrame::Command(command)))
                    .await;
                Ok(())
            }
            // A malformed dispatch call: recoverable, nothing sent.
            Publish::Reject(message) => Err(StageError::new(message)),
        }
    }
}

/// Translate one tool call. A matched, well-formed call is consumed and
/// re-emitted by `perform` after the send succeeds; a malformed dispatch call
/// forwards (staying observable) alongside its reject. An unknown tool name
/// forwards with no effect — Dispatch ignores it.
fn translate(call: &ToolCall) -> Decision<Publish> {
    let parsed = match &*call.name {
        "dispatch_task" => parse_dispatch_task(call),
        "update_task" => parse_update_task(call),
        _ => return Decision::forward(),
    };
    match parsed {
        Ok(command) => Decision::drop().emit(Publish::Command {
            command,
            call: call.clone(),
        }),
        Err(message) => Decision::forward().emit(Publish::Reject(message)),
    }
}

/// `dispatch_task` arguments: `{ "task": string, "context": string | null }`.
#[derive(Debug, Deserialize)]
struct DispatchTaskArgs {
    task: String,
    #[serde(default)]
    context: Option<String>,
}

/// `update_task` arguments: `{ "task_id": string, "message": string }`.
#[derive(Debug, Deserialize)]
struct UpdateTaskArgs {
    task_id: String,
    message: String,
}

fn parse_dispatch_task(call: &ToolCall) -> Result<DispatchCommand, Arc<str>> {
    let args: DispatchTaskArgs = serde_json::from_str(&call.arguments_json)
        .map_err(|e| Arc::<str>::from(format!("malformed dispatch_task arguments: {e}")))?;
    // TODO: try a simplified dispatch_task signature that drops `tool_call_id`
    // and `context` — just `task`.
    Ok(DispatchCommand::Create {
        tool_call_id: call.id.clone(),
        task: Arc::from(args.task),
        context: args.context.map(Arc::from),
    })
}

fn parse_update_task(call: &ToolCall) -> Result<DispatchCommand, Arc<str>> {
    let args: UpdateTaskArgs = serde_json::from_str(&call.arguments_json)
        .map_err(|e| Arc::<str>::from(format!("malformed update_task arguments: {e}")))?;
    Ok(DispatchCommand::Update {
        tool_call_id: call.id.clone(),
        task_id: Arc::from(args.task_id),
        message: Arc::from(args.message),
    })
}
