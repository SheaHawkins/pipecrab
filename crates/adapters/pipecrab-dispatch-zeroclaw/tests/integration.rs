//! Scripted flows over a mock ZeroClaw gateway (wiremock), driving the real
//! [`ZeroclawSource`] / [`ZeroclawSink`] through the transport traits.
//!
//! `next_event` / `send_command` / `cancel` are the exact surface the dispatch
//! ingress and egress stages consume, so exercising them here is exercising
//! the transport as the framework would. The `Dispatch::new(..).into_stages()`
//! pipeline round trip lives in `pipeline.rs`.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{
    config_for, create, duplicate, gateway_error, next, recv_until, reply, task_of, update,
};
use pipecrab_core::{DispatchCommand, DispatchEvent};
use pipecrab_dispatch::{DispatchSink, DispatchSource};
use pipecrab_dispatch_zeroclaw::{TaskId, ZeroclawConfig, connect};
use url::Url;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

// --- Tests -------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_yields_accepted_then_completion_and_fills_the_ring() {
    let server = MockServer::start().await;
    // The request must carry the pairing token and the tool_call_id as its
    // X-Idempotency-Key.
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .and(header("Authorization", "Bearer secret-token"))
        .and(header("X-Idempotency-Key", "call-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(reply("all done")))
        .expect(1u64)
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));

    sink.send_command(create("call-1", "book a flight"))
        .await
        .expect("create is accepted");

    let accepted = next(&mut source).await.expect("accepted");
    let task_id = task_of(&accepted);
    match &accepted {
        DispatchEvent::Accepted { tool_call_id, .. } => assert_eq!(&**tool_call_id, "call-1"),
        other => panic!("expected Accepted, got {other:?}"),
    }

    match next(&mut source).await.expect("completion") {
        DispatchEvent::Completion { message, .. } => assert_eq!(&*message, "all done"),
        other => panic!("expected Completion, got {other:?}"),
    }

    // The session key the gateway correlates on *is* the task id handed to the
    // model, and the body is the webhook contract's `{"message": ...}`.
    let request = &server.received_requests().await.unwrap()[0];
    assert_eq!(
        request.headers.get("X-Session-Id").unwrap(),
        task_id.as_str()
    );
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body, serde_json::json!({ "message": "book a flight" }));

    // The sink-side ring holds exactly what was emitted, in order.
    let ring = sink.task_events(&task_id).expect("known task");
    let kinds: Vec<&str> = ring
        .iter()
        .map(|(_, e)| match e {
            DispatchEvent::Accepted { .. } => "accepted",
            DispatchEvent::Completion { .. } => "completion",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["accepted", "completion"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn context_rides_the_message_under_a_label() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .respond_with(ResponseTemplate::new(200).set_body_json(reply("ok")))
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(DispatchCommand::Create {
        tool_call_id: Arc::from("call-1"),
        task: Arc::from("book a flight"),
        context: Some(Arc::from("window seat")),
    })
    .await
    .unwrap();
    let _ = recv_until(&mut source, |e| {
        matches!(e, DispatchEvent::Completion { .. })
    })
    .await;

    let body: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    let message = body["message"].as_str().expect("a message string");
    assert!(message.contains("book a flight"), "message: {message}");
    assert!(
        message.contains("window seat"),
        "message carries context: {message}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unpaired_gateway_fails_the_task_finally() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_json(gateway_error("Unauthorized — pair first via POST /pair")),
        )
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(create("call-1", "doomed")).await.unwrap();
    let _ = next(&mut source).await.expect("accepted");

    // Acceptance was optimistic, so the refusal arrives as this task's
    // Failure — retrying an auth failure verbatim cannot help.
    match next(&mut source).await.expect("failure") {
        DispatchEvent::Failure {
            message, retryable, ..
        } => {
            assert!(!retryable);
            assert!(message.contains("401"), "message: {message}");
            assert!(message.contains("pair"), "message: {message}");
        }
        other => panic!("expected Failure, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rate_limiting_and_provider_errors_are_retryable() {
    for (status, body) in [
        (429, gateway_error("Too many webhook requests")),
        (500, gateway_error("LLM request failed")),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/webhook"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(&server)
            .await;

        let (mut source, sink) = connect(config_for(&server));
        sink.send_command(create("call-1", "bounced"))
            .await
            .unwrap();
        let _ = next(&mut source).await.expect("accepted");

        match next(&mut source).await.expect("failure") {
            DispatchEvent::Failure { retryable, .. } => {
                assert!(retryable, "{status} must be retryable");
            }
            other => panic!("expected Failure for {status}, got {other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unconfigured_gateway_is_a_final_failure() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .respond_with(ResponseTemplate::new(503).set_body_json(
            serde_json::json!({ "error": "needs_quickstart", "url": "/quickstart" }),
        ))
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(create("call-1", "no model"))
        .await
        .unwrap();
    let _ = next(&mut source).await.expect("accepted");

    match next(&mut source).await.expect("failure") {
        DispatchEvent::Failure {
            message, retryable, ..
        } => {
            assert!(!retryable, "an unconfigured gateway will not fix itself");
            assert!(message.contains("needs_quickstart"), "message: {message}");
        }
        other => panic!("expected Failure, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_duplicate_dedupe_reply_fails_retryably() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .respond_with(ResponseTemplate::new(200).set_body_json(duplicate()))
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(create("call-1", "again?")).await.unwrap();
    let _ = next(&mut source).await.expect("accepted");

    // The gateway deduped the key without re-running the turn — and without
    // the original reply, so this cannot be a Completion.
    match next(&mut source).await.expect("failure") {
        DispatchEvent::Failure {
            message, retryable, ..
        } => {
            assert!(retryable);
            assert!(message.contains("idempotency"), "message: {message}");
        }
        other => panic!("expected Failure, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_unknown_task_is_rejected_without_http() {
    let server = MockServer::start().await;
    let (mut source, sink) = connect(config_for(&server));

    sink.send_command(DispatchCommand::Update {
        tool_call_id: Arc::from("call-u"),
        task_id: Arc::from("pc-nonexistent"),
        message: Arc::from("hello?"),
    })
    .await
    .expect("ok at the sink");

    match next(&mut source).await.expect("rejected") {
        DispatchEvent::Rejected { message, .. } => assert_eq!(&*message, "unknown task_id"),
        other => panic!("expected Rejected, got {other:?}"),
    }
    // An unknown task is rejected before any request is made.
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "no HTTP for an unknown task_id"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_while_running_is_rejected_and_posts_nothing_new() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(reply("eventually"))
                .set_delay(Duration::from_millis(300)),
        )
        .expect(1u64) // only the create posts; the mid-turn update must not.
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(create("call-1", "long errand"))
        .await
        .unwrap();
    let task_id = task_of(&next(&mut source).await.expect("accepted"));

    // The turn is still executing here (the mock is sitting on its delay).
    sink.send_command(update("call-2", &task_id, "hurry up"))
        .await
        .unwrap();

    match next(&mut source).await.expect("rejected") {
        DispatchEvent::Rejected {
            tool_call_id,
            message,
        } => {
            assert_eq!(&*tool_call_id, "call-2");
            assert!(message.contains("still running"), "message: {message}");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }

    // The original turn is unaffected by the rejected update.
    match next(&mut source).await.expect("completion") {
        DispatchEvent::Completion { message, .. } => assert_eq!(&*message, "eventually"),
        other => panic!("expected Completion, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_after_completion_chains_a_follow_up_in_the_same_session() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .and(header("X-Idempotency-Key", "call-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(reply("first answer")))
        .expect(1u64)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .and(header("X-Idempotency-Key", "call-2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(reply("second answer")))
        .expect(1u64)
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(create("call-1", "first")).await.unwrap();
    let task_id = task_of(&next(&mut source).await.expect("accepted"));

    // Drain to the first completion so the task is idle before the update.
    match next(&mut source).await.expect("first completion") {
        DispatchEvent::Completion { message, .. } => assert_eq!(&*message, "first answer"),
        other => panic!("expected Completion, got {other:?}"),
    }

    sink.send_command(update("call-2", &task_id, "again"))
        .await
        .unwrap();

    match next(&mut source).await.expect("second completion") {
        DispatchEvent::Completion {
            task_id: t,
            message,
        } => {
            assert_eq!(&*message, "second answer");
            // The follow-up reports under the very same task identity.
            assert_eq!(&*t, task_id.as_str());
        }
        other => panic!("expected Completion, got {other:?}"),
    }

    // Both posts ride the same session id — that is the whole chaining
    // mechanism, since the gateway's session memory carries the conversation.
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert_eq!(
            request.headers.get("X-Session-Id").unwrap(),
            task_id.as_str()
        );
    }
    let follow_up: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(
        follow_up,
        serde_json::json!({ "message": "again" }),
        "no history is replayed — the session carries it server-side"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_closes_the_source_issues_no_http_and_is_idempotent() {
    let server = MockServer::start().await;
    let (mut source, sink) = connect(config_for(&server));

    source.cancel();

    // The source closes gracefully.
    let closed = tokio::time::timeout(Duration::from_secs(2), source.next_event())
        .await
        .expect("closes promptly")
        .expect("no error on close");
    assert!(closed.is_none(), "cancel closes the source (Ok(None))");

    // Idempotent.
    source.cancel();

    // A send after cancel is a recoverable error, and issues no HTTP.
    let err = sink
        .send_command(create("call-late", "too late"))
        .await
        .expect_err("send after cancel errors");
    assert!(!err.fatal, "post-cancel send is recoverable");

    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "cancel and post-cancel sends touch no remote turns"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_abandons_an_in_flight_turn_without_a_failure_event() {
    let server = MockServer::start().await;
    // A turn that outlives the whole test unless abandoned.
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(reply("too late"))
                .set_delay(Duration::from_secs(60)),
        )
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(create("call-1", "slow errand"))
        .await
        .unwrap();
    let _ = next(&mut source).await.expect("accepted");

    source.cancel();

    // The source closes promptly — the worker abandons its request instead of
    // waiting out the turn, and emits nothing into the closed channel.
    let closed = tokio::time::timeout(Duration::from_secs(2), source.next_event())
        .await
        .expect("closes promptly, not after the 60s turn")
        .expect("no error on close");
    assert!(closed.is_none(), "no Failure precedes the close");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn next_event_is_cancellation_safe() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .respond_with(ResponseTemplate::new(200).set_body_json(reply("ok")))
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));

    // Poll next_event with nothing available, then drop the unresolved future.
    let dropped = tokio::time::timeout(Duration::from_millis(60), source.next_event()).await;
    assert!(
        dropped.is_err(),
        "no event yet, so the future is dropped mid-await"
    );

    // Now produce an event; the next poll must still receive it — nothing lost.
    sink.send_command(create("call-1", "work")).await.unwrap();
    match next(&mut source)
        .await
        .expect("accepted survives the dropped poll")
    {
        DispatchEvent::Accepted { tool_call_id, .. } => assert_eq!(&*tool_call_id, "call-1"),
        other => panic!("expected Accepted, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn optional_credentials_and_agent_shape_the_request() {
    let server = MockServer::start().await;
    // A tokenless config against an open gateway, with a secret and an agent.
    Mock::given(method("POST"))
        .and(path("/webhook"))
        .and(header("X-Webhook-Secret", "hook-secret"))
        .and(query_param("agent", "researcher"))
        .respond_with(ResponseTemplate::new(200).set_body_json(reply("ok")))
        .expect(1u64)
        .mount(&server)
        .await;

    let config = ZeroclawConfig::new()
        .with_base_url(Url::parse(&server.uri()).unwrap())
        .with_webhook_secret("hook-secret")
        .with_agent("researcher")
        .with_request_timeout(Duration::from_secs(2));
    let (mut source, sink) = connect(config);
    sink.send_command(create("call-1", "look it up"))
        .await
        .unwrap();
    let _ = recv_until(&mut source, |e| {
        matches!(e, DispatchEvent::Completion { .. })
    })
    .await;

    // No token configured → no Authorization header sent.
    let request = &server.received_requests().await.unwrap()[0];
    assert!(request.headers.get("Authorization").is_none());
}

#[test]
fn credentials_are_redacted_from_config_debug() {
    let config = ZeroclawConfig::new()
        .with_token("super-secret-token")
        .with_webhook_secret("super-secret-hook");
    let rendered = format!("{config:?}");
    assert!(
        !rendered.contains("super-secret"),
        "credentials must not appear: {rendered}"
    );
    assert!(rendered.contains("<redacted>"));
}

#[test]
fn task_id_round_trips_through_its_conversions() {
    let id = TaskId::from("pc-roundtrip");
    assert_eq!(id.as_str(), "pc-roundtrip");
    assert_eq!(id.to_string(), "pc-roundtrip");
}
