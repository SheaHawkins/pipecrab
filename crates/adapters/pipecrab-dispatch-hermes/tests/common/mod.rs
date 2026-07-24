//! Shared wiremock scaffolding for the Hermes adapter integration tests: a
//! sequence responder, run-body constructors, a config pointed at the mock, and
//! a couple of event-reading helpers.
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use pipecrab_core::{DispatchCommand, DispatchEvent};
use pipecrab_dispatch::DispatchSource;
use pipecrab_dispatch_hermes::{HermesConfig, HermesSource, TaskId};
use serde_json::{Value, json};
use url::Url;
use wiremock::{MockServer, Request, Respond, ResponseTemplate};

/// A responder that walks a fixed list of responses, sticking on the last once
/// exhausted — enough to script `started → running → completed` progressions and
/// incrementing create responses without stateful mock scenarios.
pub struct Sequence {
    /// `(status, body)` pairs, walked in order.
    steps: Vec<(u16, Value)>,
    calls: Arc<AtomicUsize>,
}

impl Sequence {
    /// A sequence of bodies that all share one status code.
    pub fn new(status: u16, bodies: Vec<Value>) -> Self {
        Self::new_with_statuses(bodies.into_iter().map(|b| (status, b)).collect())
    }

    /// A sequence whose status code varies per step — for scripting transient
    /// failures ahead of a success.
    pub fn new_with_statuses(steps: Vec<(u16, Value)>) -> Self {
        assert!(!steps.is_empty(), "a Sequence needs at least one response");
        Self {
            steps,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Respond for Sequence {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let i = self
            .calls
            .fetch_add(1, Ordering::SeqCst)
            .min(self.steps.len() - 1);
        let (status, body) = &self.steps[i];
        ResponseTemplate::new(*status).set_body_json(body.clone())
    }
}

pub fn started(run: &str) -> Value {
    json!({ "run_id": run, "status": "started" })
}

pub fn run_status(run: &str, status: &str) -> Value {
    json!({ "object": "hermes.run", "run_id": run, "status": status, "session_id": "pc-x", "model": "hermes-agent" })
}

pub fn completed(run: &str, output: &str) -> Value {
    json!({ "object": "hermes.run", "run_id": run, "status": "completed", "output": output, "session_id": "pc-x" })
}

pub fn config_for(server: &MockServer) -> HermesConfig {
    HermesConfig::new("secret-token")
        .with_base_url(Url::parse(&server.uri()).unwrap())
        // Tight timings keep the tests quick and deterministic.
        .with_poll_interval(Duration::from_millis(40))
        .with_request_timeout(Duration::from_secs(2))
}

/// Await the next event within a generous budget, panicking on timeout or error.
pub async fn next(source: &mut HermesSource) -> Option<DispatchEvent> {
    tokio::time::timeout(Duration::from_secs(5), source.next_event())
        .await
        .expect("an event within the time budget")
        .expect("the source must not error")
}

/// Read events until one satisfies `pred`, discarding any that precede it.
pub async fn recv_until(
    source: &mut HermesSource,
    pred: impl Fn(&DispatchEvent) -> bool,
) -> DispatchEvent {
    loop {
        let event = next(source).await.expect("the source stayed open");
        if pred(&event) {
            return event;
        }
    }
}

pub fn create(tool_call_id: &str, task: &str) -> DispatchCommand {
    DispatchCommand::Create {
        tool_call_id: Arc::from(tool_call_id),
        task: Arc::from(task),
        context: None,
    }
}

pub fn update(tool_call_id: &str, task_id: &TaskId, message: &str) -> DispatchCommand {
    DispatchCommand::Update {
        tool_call_id: Arc::from(tool_call_id),
        task_id: Arc::from(task_id.as_str()),
        message: Arc::from(message),
    }
}

/// The [`TaskId`] carried by an `Accepted` event.
pub fn task_of(event: &DispatchEvent) -> TaskId {
    match event {
        DispatchEvent::Accepted { task_id, .. } => TaskId::from(task_id.clone()),
        other => panic!("expected Accepted, got {other:?}"),
    }
}
