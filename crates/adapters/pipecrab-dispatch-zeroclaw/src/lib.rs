//! A [`pipecrab_dispatch`] transport backed by a **ZeroClaw** gateway's
//! `/webhook` endpoint.
//!
//! [`connect`] hands back a [`ZeroclawSource`] / [`ZeroclawSink`] pair that
//! plugs straight into [`Dispatch::new`](pipecrab_dispatch::Dispatch::new): the
//! source feeds the ingress stage, the sink drives the egress stage. Each
//! accepted command spawns a detached tokio worker that performs one
//! `POST /webhook` and projects its outcome onto a
//! [`DispatchEvent`](pipecrab_core::DispatchEvent).
//!
//! # Model
//!
//! `POST /webhook` is **synchronous**: the gateway answers only once the whole
//! agent turn has run, with `{"response": ..., "model": ...}`. There are no run
//! ids and no status endpoint to poll, which shapes the adapter differently
//! from `pipecrab-dispatch-hermes`:
//!
//! * A `dispatch_task` tool call mints a stable [`TaskId`] (`pc-{uuid}`), sends
//!   it as the request's `X-Session-Id`, and emits
//!   [`Accepted`](pipecrab_core::DispatchEvent::Accepted) *as the request
//!   departs* — there is no cheap acceptance signal to await, since any gateway
//!   answer takes a whole agent turn to arrive. A gateway-level refusal (401,
//!   429, ...) therefore surfaces as a prompt [`Failure`], not a `Rejected`.
//! * The worker maps the eventual reply: a 2xx `response` → `Completion`, a 4xx
//!   → non-retryable `Failure`, a 429 / 5xx / transport error → retryable
//!   `Failure`, and a `needs_quickstart` 503 (gateway booted without a model) →
//!   non-retryable `Failure`.
//! * `update_task` posts a follow-up with the task's prior exchanges replayed
//!   ahead of it — the webhook is stateless, so the adapter is what makes a
//!   task a conversation. It works only between turns: the webhook API has no
//!   way to deliver input to an executing turn, so an update mid-turn is
//!   rejected rather than silently deferred.
//! * This adapter never emits [`Progress`](pipecrab_core::DispatchEvent::Progress)
//!   or [`Question`](pipecrab_core::DispatchEvent::Question) — one request is
//!   one opaque turn, with no intermediate state to observe. An agent's
//!   question surfaces as `Completion` text, which the dispatch update flow
//!   already handles.
//! * [`cancel`](pipecrab_dispatch::DispatchSource::cancel) closes the source
//!   and abandons in-flight requests, deliberately leaving the gateway to
//!   finish its turns — external task state outlives the turn.
//!
//! # Native only
//!
//! Like `pipecrab-audio-cpal`, this crate depends on `tokio` (the per-command
//! workers) and `reqwest` (HTTP), so it is not part of the wasm portability
//! matrix.
//!
//! # Gateway behavior, as observed against a live gateway (2026-08)
//!
//! * The gateway listens on port `42617` by default. Auth is a pairing bearer
//!   token (`Authorization: Bearer <token>`, obtained via `POST /pair`), plus
//!   an optional `X-Webhook-Secret` layer; either may be disabled
//!   gateway-side, so both are optional in [`ZeroclawConfig`].
//! * The body is `{"message": "..."}`. A 2xx reply is
//!   `{"response": ..., "model": ...}`; errors are `{"error": ...}` envelopes
//!   (400 bad JSON / unknown `?agent=` alias, 401 unpaired, 429 rate limited
//!   with `retry_after`, 500 provider failure, 503 `needs_quickstart`).
//! * The webhook is **stateless**: each POST is its own conversation, and the
//!   gateway keeps no history between them. `X-Session-Id` is accepted (it sits
//!   in the gateway's permitted-header list) but is read by no handler on this
//!   path, so it correlates nothing — the adapter still sends the task id there
//!   for gateways that do honor it, and replays its own transcript so a
//!   follow-up turn is understood either way. The gateway's `/api/sessions`
//!   sessions are ACP sessions; a webhook post creates none.
//! * `X-Idempotency-Key` **is** honored: a repost with a seen key answers
//!   `{"status": "duplicate", "idempotent": true, ...}` without re-running the
//!   turn — and without the original reply, which is why the adapter maps it
//!   to a retryable `Failure`. The tool call id is sent as the key.
//! * An optional `?agent=<alias>` query selects a configured agent; see
//!   [`ZeroclawConfig::with_agent`].
//!
//! # Wiring
//!
//! ```no_run
//! use pipecrab_dispatch::Dispatch;
//! use pipecrab_dispatch_zeroclaw::{ZeroclawConfig, connect};
//!
//! # async fn wire() {
//! // `send_command` spawns its worker with tokio::spawn, so drive the
//! // pipeline inside a tokio runtime.
//! let config = ZeroclawConfig::new().with_token("PAIRING_TOKEN");
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
//!
//! [`Failure`]: pipecrab_core::DispatchEvent::Failure
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::Arc;

mod classify;
mod client;
mod config;
mod inner;
mod sink;
mod source;
mod task;

pub use config::ZeroclawConfig;
pub use sink::ZeroclawSink;
pub use source::ZeroclawSource;
pub use task::TaskId;

use client::ZeroclawClient;
use inner::Inner;

/// Connect to a ZeroClaw gateway, returning the transport pair.
///
/// Spawns nothing itself — each command spawns its own worker from
/// [`send_command`](pipecrab_dispatch::DispatchSink::send_command), which must
/// therefore run inside a tokio runtime.
pub fn connect(config: ZeroclawConfig) -> (ZeroclawSource, ZeroclawSink) {
    let client = ZeroclawClient::new(&config);
    let (inner, events) = Inner::new(client);
    let inner = Arc::new(inner);

    let source = ZeroclawSource::new(Arc::clone(&inner), events);
    let sink = ZeroclawSink::new(inner);
    (source, sink)
}
