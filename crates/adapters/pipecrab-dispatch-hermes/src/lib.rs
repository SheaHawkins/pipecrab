//! A [`pipecrab-dispatch`] transport backed by the **Hermes Agent** gateway's
//! runs API (NousResearch).
//!
//! [`connect`] hands back a [`HermesSource`] / [`HermesSink`] pair that plugs
//! straight into [`Dispatch::new`](pipecrab_dispatch::Dispatch::new): the source
//! feeds the ingress stage, the sink drives the egress stage. Behind them a
//! detached tokio task polls every active run on an interval and projects each
//! transition onto a [`DispatchEvent`](pipecrab_core::DispatchEvent).
//!
//! # Model
//!
//! * A `dispatch_task` tool call becomes `POST /v1/runs`. The adapter mints a
//!   stable [`TaskId`] (`pc-{uuid}`), sends it as the run's `session_id`, and
//!   hands it back as the accepted task's id — so the id the model holds is
//!   stable across the follow-up runs an `update_task` chains into the same
//!   session. A [`TaskId`] is **not** a Hermes `run_id`: one task, many runs.
//! * The poller `GET`s each active run and maps its status:
//!   `completed` → `Completion`, `failed` → non-retryable `Failure`,
//!   `cancelled` → retryable `Failure`, a 404 on a tracked run →
//!   expired-state `Failure`, and any other status change → a deduped
//!   `Progress`. Network blips and 5xx/401 responses are retried silently.
//! * `update_task` chains a *new* run and replays the task's turns as
//!   `conversation_history`, since a `session_id` correlates runs without
//!   carrying any conversation. It works only between runs: the runs API has no
//!   endpoint that delivers input to an executing run, so an update mid-run is
//!   rejected rather than silently deferred.
//! * A failed create/update surfaces as a `Rejected` event (respond-worthy to
//!   the model), never a pipeline error.
//!
//! # Scope
//!
//! v1 is **polling only** — no SSE, no approvals, no `/stop`. Consequently this
//! adapter never emits [`DispatchEvent::Question`]; an agent's question surfaces
//! as `Completion` text, which the dispatch update flow already handles.
//! [`cancel`](pipecrab_dispatch::DispatchSource::cancel) stops the poller and
//! closes the source but deliberately leaves remote runs running — external task
//! state outlives the turn.
//!
//! # Native only
//!
//! Like `pipecrab-audio-cpal`, this crate depends on `tokio` (the poller task)
//! and `reqwest` (HTTP), so it is not part of the wasm portability matrix.
//!
//! # Gateway behavior, as observed on 0.18.2
//!
//! * Statuses run `started` → `running` → `completed` / `cancelled`, with
//!   `stopping` transitional. Unrecognized statuses are treated as in-flight, so
//!   a gateway that adds one stays compatible.
//! * `POST /v1/runs` answers `202` with `{run_id, status}`; a `session_id` the
//!   gateway has not seen is created implicitly, and posting into a session with
//!   a run already executing is accepted rather than refused (this adapter still
//!   rejects that case — racing two runs in one session is undefined).
//! * A completed run's body carries `output` and `usage`; a terminal run is
//!   retained well beyond the default 1s poll interval. A 404 is an
//!   OpenAI-shaped error envelope.
//! * A `session_id` does **not** chain conversation: a second run in the same
//!   session starts blank, and `previous_response_id` set to a prior `run_id`
//!   does not help either. Only an explicit `conversation_history` array carries
//!   context, which is why follow-up runs replay one.
//! * `Idempotency-Key` is **not** honored on `/v1/runs`: a repost with the same
//!   key yields a new run. It is still sent, but nothing depends on it.
//! * No error-detail field name is documented for a failed run, so the adapter
//!   probes `error_message`, then `error` / `last_error` as a string or an
//!   object with a `message`.
//!
//! # Wiring
//!
//! ```no_run
//! use std::time::Duration;
//! use pipecrab_dispatch::Dispatch;
//! use pipecrab_dispatch_hermes::{HermesConfig, connect};
//!
//! # async fn wire() {
//! // Inside a tokio runtime — `connect` spawns the poller task.
//! let config = HermesConfig::new("API_SERVER_KEY")
//!     .with_poll_interval(Duration::from_secs(1));
//! let (source, sink) = connect(config);
//!
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
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::Arc;

mod classify;
mod client;
mod config;
mod inner;
mod poller;
mod sink;
mod source;
mod task;

pub use config::HermesConfig;
pub use sink::HermesSink;
pub use source::HermesSource;
pub use task::TaskId;

use client::HermesClient;
use inner::Inner;

/// Connect to a Hermes gateway, returning the transport pair and spawning the
/// background poller.
///
/// # Panics / runtime
///
/// Must be called inside a tokio runtime: it spawns the poller with
/// [`tokio::spawn`], which panics outside a runtime context.
pub fn connect(config: HermesConfig) -> (HermesSource, HermesSink) {
    let client = HermesClient::new(&config);
    let (inner, events) = Inner::new(client, config.poll_interval);
    let inner = Arc::new(inner);

    tokio::spawn(poller::run(Arc::clone(&inner)));

    let source = HermesSource::new(Arc::clone(&inner), events);
    let sink = HermesSink::new(inner);
    (source, sink)
}
