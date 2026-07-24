//! Scripted flows over a mock Hermes gateway (wiremock), driving the real
//! [`HermesSource`] / [`HermesSink`] through the transport traits.
//!
//! `next_event` / `send_command` / `cancel` are the exact surface the dispatch
//! ingress and egress stages consume, so exercising them here is exercising the
//! transport as the framework would. The `Dispatch::new(..).into_stages()`
//! pipeline round trip lives in `pipeline.rs`.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{
    Sequence, completed, config_for, create, next, recv_until, run_status, started, task_of, update,
};
use pipecrab_core::{DispatchCommand, DispatchEvent};
use pipecrab_dispatch::{DispatchSink, DispatchSource};
use pipecrab_dispatch_hermes::{HermesConfig, TaskId, connect};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// --- Tests -------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_yields_accepted_then_progress_then_completion_and_fills_the_ring() {
    let server = MockServer::start().await;
    // The create must carry the tool_call_id as its Idempotency-Key.
    Mock::given(method("POST"))
        .and(path("/v1/runs"))
        .and(header("Idempotency-Key", "call-1"))
        .respond_with(ResponseTemplate::new(202).set_body_json(started("run_a")))
        .expect(1u64)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/runs/run_a"))
        .respond_with(Sequence::new(
            200,
            vec![
                run_status("run_a", "running"),
                completed("run_a", "all done"),
            ],
        ))
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

    match next(&mut source).await.expect("progress") {
        DispatchEvent::Progress { message, .. } => assert_eq!(&*message, "running"),
        other => panic!("expected Progress(running), got {other:?}"),
    }
    match next(&mut source).await.expect("completion") {
        DispatchEvent::Completion { message, .. } => assert_eq!(&*message, "all done"),
        other => panic!("expected Completion, got {other:?}"),
    }

    // The sink-side ring holds exactly what was emitted, in order.
    let ring = sink.task_events(&task_id).expect("known task");
    let kinds: Vec<&str> = ring
        .iter()
        .map(|(_, e)| match e {
            DispatchEvent::Accepted { .. } => "accepted",
            DispatchEvent::Progress { .. } => "progress",
            DispatchEvent::Completion { .. } => "completion",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["accepted", "progress", "completion"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_http_failure_is_rejected_not_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/runs"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream boom"))
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));

    // A failed create is Ok(()) at the sink — the failure travels as an event.
    sink.send_command(create("call-2", "do it"))
        .await
        .expect("a rejected create is still Ok at the sink");

    match next(&mut source).await.expect("rejected") {
        DispatchEvent::Rejected {
            tool_call_id,
            message,
        } => {
            assert_eq!(&*tool_call_id, "call-2");
            assert!(
                message.contains("500"),
                "message names the status: {message}"
            );
        }
        other => panic!("expected Rejected, got {other:?}"),
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
async fn update_while_running_is_rejected_and_posts_no_new_run() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/runs"))
        .respond_with(ResponseTemplate::new(202).set_body_json(started("run_a")))
        .expect(1u64) // only the create posts; the mid-run update must not.
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/runs/run_a"))
        .respond_with(Sequence::new(200, vec![run_status("run_a", "running")]))
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(create("call-1", "long errand"))
        .await
        .unwrap();
    let task_id = task_of(&next(&mut source).await.expect("accepted"));

    // The run never terminates, so the task is still active here.
    sink.send_command(update("call-2", &task_id, "hurry up"))
        .await
        .unwrap();

    let rejected = recv_until(&mut source, |e| matches!(e, DispatchEvent::Rejected { .. })).await;
    match rejected {
        DispatchEvent::Rejected {
            tool_call_id,
            message,
        } => {
            assert_eq!(&*tool_call_id, "call-2");
            assert!(message.contains("still running"), "message: {message}");
        }
        _ => unreachable!(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_after_completion_chains_a_new_run_under_the_same_task() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/runs"))
        .respond_with(Sequence::new(202, vec![started("run_a"), started("run_b")]))
        .expect(2u64) // create + the chained update.
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/runs/run_a"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completed("run_a", "first answer")))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/runs/run_b"))
        .respond_with(Sequence::new(
            200,
            vec![
                run_status("run_b", "running"),
                completed("run_b", "second answer"),
            ],
        ))
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(create("call-1", "first")).await.unwrap();
    let task_id = task_of(&next(&mut source).await.expect("accepted"));

    // Drain to the first completion so the task is idle before the update.
    let first = recv_until(&mut source, |e| {
        matches!(e, DispatchEvent::Completion { .. })
    })
    .await;
    let first_task = match &first {
        DispatchEvent::Completion { task_id, message } => {
            assert_eq!(&**message, "first answer");
            TaskId::from(task_id.clone())
        }
        _ => unreachable!(),
    };

    sink.send_command(update("call-2", &task_id, "again"))
        .await
        .unwrap();

    let second = recv_until(&mut source, |e| {
        matches!(e, DispatchEvent::Completion { .. })
    })
    .await;
    match &second {
        DispatchEvent::Completion {
            task_id: t,
            message,
        } => {
            assert_eq!(&**message, "second answer");
            // The follow-up run reports under the very same task identity.
            assert_eq!(TaskId::from(t.clone()).as_str(), first_task.as_str());
        }
        _ => unreachable!(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn poll_404_on_a_tracked_run_is_an_expired_state_failure() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/runs"))
        .respond_with(ResponseTemplate::new(202).set_body_json(started("run_a")))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/runs/run_a"))
        .respond_with(ResponseTemplate::new(404).set_body_json(
            json!({ "error": { "message": "Run not found", "code": "run_not_found" } }),
        ))
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(create("call-1", "vanishing task"))
        .await
        .unwrap();
    let _ = next(&mut source).await.expect("accepted");

    match next(&mut source).await.expect("failure") {
        DispatchEvent::Failure {
            message, retryable, ..
        } => {
            assert_eq!(&*message, "run state expired before it was observed");
            assert!(!retryable);
        }
        other => panic!("expected Failure, got {other:?}"),
    }

    // The run is cleared, so the failure fires once — no repeat on the next tick.
    let repeat = tokio::time::timeout(Duration::from_millis(200), source.next_event()).await;
    assert!(repeat.is_err(), "the expired-state failure must not repeat");
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
        "cancel and post-cancel sends touch no remote runs"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn next_event_is_cancellation_safe() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/runs"))
        .respond_with(ResponseTemplate::new(202).set_body_json(started("run_a")))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/runs/run_a"))
        .respond_with(Sequence::new(200, vec![run_status("run_a", "running")]))
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
async fn the_create_body_carries_the_minted_task_id_as_session_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/runs"))
        .respond_with(ResponseTemplate::new(202).set_body_json(started("run_a")))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/runs/run_a"))
        .respond_with(Sequence::new(200, vec![run_status("run_a", "running")]))
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
    let task_id = task_of(&next(&mut source).await.expect("accepted"));

    let requests = server.received_requests().await.unwrap();
    let create_body: serde_json::Value = requests
        .iter()
        .find(|r| r.method == wiremock::http::Method::POST)
        .map(|r| serde_json::from_slice(&r.body).expect("a JSON body"))
        .expect("a create request was made");

    // The session key Hermes correlates on *is* the task id handed to the model.
    assert_eq!(create_body["session_id"], task_id.as_str());
    // The context is composed into the run input under a label.
    let input = create_body["input"].as_str().expect("an input string");
    assert!(input.contains("book a flight"), "input: {input}");
    assert!(
        input.contains("window seat"),
        "input carries context: {input}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn poll_side_network_errors_are_retried_silently() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/runs"))
        .respond_with(ResponseTemplate::new(202).set_body_json(started("run_a")))
        .mount(&server)
        .await;
    // Two 500s, then a real completion: the blips must not fail the task.
    Mock::given(method("GET"))
        .and(path("/v1/runs/run_a"))
        .respond_with(Sequence::new_with_statuses(vec![
            (500, json!({ "error": "gateway blip" })),
            (503, json!({ "error": "still blipping" })),
            (200, completed("run_a", "survived the blips")),
        ]))
        .mount(&server)
        .await;

    let (mut source, sink) = connect(config_for(&server));
    sink.send_command(create("call-1", "resilient task"))
        .await
        .unwrap();
    let _ = next(&mut source).await.expect("accepted");

    // The very next event is the completion — the 5xx ticks emitted nothing.
    match next(&mut source).await.expect("completion") {
        DispatchEvent::Completion { message, .. } => {
            assert_eq!(&*message, "survived the blips");
        }
        other => panic!("a transient poll error must emit nothing, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_event_channel_pauses_the_poller_without_losing_events() {
    // More distinct statuses than the channel holds (capacity 64), so the poller
    // must block on a full channel rather than drop or skip.
    const STATUSES: usize = 90;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/runs"))
        .respond_with(ResponseTemplate::new(202).set_body_json(started("run_a")))
        .mount(&server)
        .await;
    let mut bodies: Vec<serde_json::Value> = (0..STATUSES)
        .map(|i| run_status("run_a", &format!("step-{i}")))
        .collect();
    bodies.push(completed("run_a", "finished"));
    Mock::given(method("GET"))
        .and(path("/v1/runs/run_a"))
        .respond_with(Sequence::new(200, bodies))
        .mount(&server)
        .await;

    let config = config_for(&server).with_poll_interval(Duration::from_millis(1));
    let (mut source, sink) = connect(config);
    sink.send_command(create("call-1", "chatty task"))
        .await
        .unwrap();
    let _ = next(&mut source).await.expect("accepted");

    // Let the poller run far past the channel's capacity while nothing drains.
    tokio::time::sleep(Duration::from_millis(400)).await;

    // Now drain: every status must arrive, in order, with none skipped.
    let mut seen = Vec::new();
    loop {
        match next(&mut source).await.expect("the source stayed open") {
            DispatchEvent::Progress { message, .. } => seen.push(message.to_string()),
            DispatchEvent::Completion { message, .. } => {
                assert_eq!(&*message, "finished");
                break;
            }
            other => panic!("unexpected event {other:?}"),
        }
    }

    let expected: Vec<String> = (0..STATUSES).map(|i| format!("step-{i}")).collect();
    assert_eq!(
        seen, expected,
        "a full channel must pause the poller, never skip or drop a transition"
    );
}

#[test]
fn api_key_is_redacted_from_config_debug() {
    let config = HermesConfig::new("super-secret-key");
    let rendered = format!("{config:?}");
    assert!(
        !rendered.contains("super-secret-key"),
        "key must not appear: {rendered}"
    );
    assert!(rendered.contains("<redacted>"));
}
