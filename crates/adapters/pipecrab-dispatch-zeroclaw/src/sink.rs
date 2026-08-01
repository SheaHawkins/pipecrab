//! [`ZeroclawSink`]: the send half. Translates a [`DispatchCommand`] into a
//! webhook turn and emits the resulting [`Accepted`](DispatchEvent::Accepted) /
//! [`Rejected`](DispatchEvent::Rejected) onto the shared event channel; the
//! turn's terminal event arrives later from the detached worker.

use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use pipecrab_core::{DispatchCommand, DispatchEvent};
use pipecrab_dispatch::{DispatchError, DispatchSink};

use crate::inner::Inner;
use crate::task::TaskId;

/// The send half of the ZeroClaw transport, a shared `Arc<Inner>` handle.
/// Cloneable so it can be held by an egress stage and still be inspected
/// elsewhere.
#[derive(Clone)]
pub struct ZeroclawSink {
    inner: Arc<Inner>,
}

impl ZeroclawSink {
    pub(crate) fn new(inner: Arc<Inner>) -> Self {
        Self { inner }
    }

    /// A cloned, oldest-first snapshot of every event emitted for `task_id`,
    /// or `None` if no such task exists.
    ///
    /// This is a sink-side convenience for inspection and debugging, bounded
    /// to the last few dozen events. It is *not* the system of record: the
    /// authoritative event stream is the raw `DispatchFrame::Event`s ingress
    /// emits into the pipeline. Two consequences follow from that split: a
    /// create-time `Rejected` precedes the task's existence and so appears in
    /// no ring, and a `Rejected` carries no `task_id` to look one up by.
    pub fn task_events(&self, task_id: &TaskId) -> Option<Vec<(SystemTime, DispatchEvent)>> {
        self.inner.task_events(task_id)
    }
}

#[async_trait]
impl DispatchSink for ZeroclawSink {
    async fn send_command(&self, command: DispatchCommand) -> Result<(), DispatchError> {
        // A send after cancellation is recoverable adapter-internal breakage:
        // the pipeline is winding down, so no HTTP, no event.
        if self.inner.is_cancelled() {
            return Err(DispatchError::recoverable("dispatch transport cancelled"));
        }
        match command {
            DispatchCommand::Create {
                tool_call_id,
                task,
                context,
            } => self.inner.create(tool_call_id, task, context).await,
            DispatchCommand::Update {
                tool_call_id,
                task_id,
                message,
            } => self.inner.update(tool_call_id, task_id, message).await,
        }
    }
}
