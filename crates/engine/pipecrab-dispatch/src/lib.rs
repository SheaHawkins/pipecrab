//! pipecrab-dispatch: connect native `Dispatch` frames to model tool calls and
//! external asynchronous-task transports.
//!
//! This is the *facade* layer. It stitches the generic LM interface
//! ([`pipecrab_lm`]) onto the native dispatch frames ([`pipecrab_core`]) while LM
//! stays independent of dispatch. It owns the dispatch tool definitions, the
//! event-to-model projection, the active ingress stage, the tool-call-to-command
//! egress stage, and the transport capability traits — but *not* durable task
//! state or a concrete wire protocol.
//!
//! # Wiring it to `LmStage`
//!
//! The tool definitions are ordinary [`ToolDefinition`]s, so they feed
//! [`LmStage::with_tools`](pipecrab_lm::LmStage::with_tools) directly:
//!
//! ```no_run
//! # use pipecrab_dispatch::{Dispatch, DispatchSource, DispatchSink};
//! # fn wire<S: DispatchSource + 'static, K: DispatchSink + 'static>(source: S, sink: K) {
//! let dispatch = Dispatch::new(source, sink);
//! let tools = dispatch.tool_definitions();
//! let (ingress, egress) = dispatch.into_stages();
//!
//! // let lm = LmStage::with_tools(model, system_prompt, tools.iter().cloned())?;
//! // let pipeline = PipelineBuilder::new()
//! //     .stage(ingress)
//! //     .stage(lm)
//! //     .stage(egress)
//! //     .build();
//! # let _ = (tools, ingress, egress);
//! # }
//! ```
//!
//! The caller then drives the pipeline's single driver future — there is no
//! separate dispatch listener to run (see [`DispatchIngress`]).
//!
//! # `tool_call_id` vs `task_id`
//!
//! A `tool_call_id` names one model *invocation* — it exists the moment the model
//! emits a `dispatch_task` call. A `task_id` names the durable *task* and exists
//! only once the backend accepts that call ([`DispatchEvent::Accepted`]). Ingress
//! projects the acceptance into a tool result carrying the `task_id`, which is how
//! a later user answer can turn into `update_task(task_id = ..)` without forcing
//! an immediate reply.
//!
//! # The two round-trip halves
//!
//! * [`DispatchIngress`] — *active* stage: injects external
//!   [`DispatchEvent`](pipecrab_core::DispatchEvent)s into an otherwise idle
//!   pipeline, each as a raw `Dispatch` frame followed by a model projection.
//! * [`DispatchEgress`] — translator: turns `dispatch_task` / `update_task` tool
//!   calls into [`DispatchCommand`](pipecrab_core::DispatchCommand)s, publishes
//!   them to the sink, and echoes them downstream.
//!
//! Concrete transports (`pipecrab-dispatch-websocket`, `-http`, `-hermes`) are
//! later adapter crates that implement [`DispatchSource`] / [`DispatchSink`].
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::Arc;

pub mod egress;
pub mod error;
pub mod ingress;
pub mod tools;
pub mod transport;

pub use egress::{DispatchEgress, Publish};
pub use error::DispatchError;
pub use ingress::DispatchIngress;
pub use tools::{dispatch_task_definition, dispatch_tool_definitions, update_task_definition};
pub use transport::{DispatchSink, DispatchSource, DispatchTransport};

/// The provider-neutral tool definition type, re-exported so callers name one
/// crate.
pub use pipecrab_lm::ToolDefinition;

/// A small composition helper: bundle a [`DispatchSource`] and a [`DispatchSink`]
/// to hand out the tool definitions and the two pipeline stages together.
///
/// ```no_run
/// # use pipecrab_dispatch::{Dispatch, DispatchSource, DispatchSink};
/// # fn go<S: DispatchSource + 'static, K: DispatchSink + 'static>(source: S, sink: K) {
/// let dispatch = Dispatch::new(source, sink);
/// let tools = dispatch.tool_definitions();
/// let (ingress, egress) = dispatch.into_stages();
/// # let _ = (tools, ingress, egress);
/// # }
/// ```
pub struct Dispatch<S, K> {
    source: S,
    sink: K,
}

impl<S, K> Dispatch<S, K>
where
    S: DispatchSource,
    K: DispatchSink,
{
    /// Bundle a source and a sink.
    pub fn new(source: S, sink: K) -> Self {
        Self { source, sink }
    }

    /// The dispatch tool definitions to configure on an
    /// [`LmStage`](pipecrab_lm::LmStage). Cheap to call; clones an `Arc`.
    pub fn tool_definitions(&self) -> Arc<[ToolDefinition]> {
        dispatch_tool_definitions()
    }

    /// Split into the ingress and egress stages for a pipeline. The source and
    /// sink move into their respective stages.
    pub fn into_stages(self) -> (DispatchIngress<S>, DispatchEgress<K>) {
        (
            DispatchIngress::new(self.source),
            DispatchEgress::new(self.sink),
        )
    }
}
