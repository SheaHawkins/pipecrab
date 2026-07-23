//! [`DispatchEgress`]: turns model tool calls into [`DispatchCommand`]s,
//! publishes them through a [`DispatchSink`], and echoes them downstream as
//! native dispatch frames.
//!
//! Egress is pure mechanism — a per-tool-call translator. It holds no state: no
//! task map, no `task_id`s, no generation bookkeeping. Durable task state lives
//! in the backend behind the transport. Behavioral guidance (e.g. "speak an
//! acknowledgement before dispatching") is the model's job and lives in the tool
//! *descriptions*, not in a hard gate here — gating would reject a valid
//! tool-call-only generation and silently drop the user's task.

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
    /// Send this command to the sink, then emit it downstream as a native
    /// `Dispatch` frame.
    Command(DispatchCommand),
    /// A dispatch tool call whose arguments would not parse. Surfaces as a
    /// recoverable [`StageError`]; no command is sent.
    Reject(Arc<str>),
}

/// Translates `dispatch_task` / `update_task` tool calls into
/// [`DispatchCommand`]s, sends them through the [`DispatchSink`], and forwards
/// them downstream as native dispatch frames — leaving the original
/// [`ModelFrame::ToolCall`] in the stream for other observers. Unknown tool calls
/// pass through untouched.
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
            Publish::Command(command) => {
                // Publish to the transport first, then echo the native command
                // downstream so it is observable without instrumenting the sink.
                self.sink.send_command(command.clone()).await?;
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

/// Translate one tool call. The call always forwards (it stays visible to
/// downstream observers); the emitted effect, if any, drives the sink. An
/// unknown tool name forwards with no effect — Dispatch ignores it.
fn translate(call: &ToolCall) -> Decision<Publish> {
    let parsed = match &*call.name {
        "dispatch_task" => parse_dispatch_task(call),
        "update_task" => parse_update_task(call),
        _ => return Decision::forward(),
    };
    match parsed {
        Ok(command) => Decision::forward().emit(Publish::Command(command)),
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
