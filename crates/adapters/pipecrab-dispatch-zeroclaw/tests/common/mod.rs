//! Shared wiremock scaffolding for the ZeroClaw adapter integration tests: a
//! config pointed at the mock, reply-body constructors, and a couple of
//! event-reading helpers.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use pipecrab_core::{DispatchCommand, DispatchEvent};
use pipecrab_dispatch::DispatchSource;
use pipecrab_dispatch_zeroclaw::{TaskId, ZeroclawConfig, ZeroclawSource};
use serde_json::{Value, json};
use url::Url;
use wiremock::MockServer;

pub fn reply(text: &str) -> Value {
    json!({ "response": text, "model": "mock-model" })
}

pub fn gateway_error(text: &str) -> Value {
    json!({ "error": text })
}

pub fn duplicate() -> Value {
    json!({
        "status": "duplicate",
        "idempotent": true,
        "message": "Request already processed for this idempotency key"
    })
}

pub fn config_for(server: &MockServer) -> ZeroclawConfig {
    ZeroclawConfig::new()
        .with_token("secret-token")
        .with_base_url(Url::parse(&server.uri()).unwrap())
        // Tight timing keeps the tests quick and deterministic.
        .with_request_timeout(Duration::from_secs(2))
}

/// Await the next event within a generous budget, panicking on timeout or error.
pub async fn next(source: &mut ZeroclawSource) -> Option<DispatchEvent> {
    tokio::time::timeout(Duration::from_secs(5), source.next_event())
        .await
        .expect("an event within the time budget")
        .expect("the source must not error")
}

/// Read events until one satisfies `pred`, discarding any that precede it.
pub async fn recv_until(
    source: &mut ZeroclawSource,
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
