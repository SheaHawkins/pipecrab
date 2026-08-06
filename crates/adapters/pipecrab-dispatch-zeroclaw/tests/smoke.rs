//! Live smoke test against a real `zeroclaw gateway`. Ignored by default — it
//! needs a reachable gateway and burns real model time.
//!
//! ```sh
//! ZEROCLAW_SMOKE_URL=http://127.0.0.1:42617 \
//! ZEROCLAW_SMOKE_TOKEN=$PAIRING_TOKEN \
//!   cargo test -p pipecrab-dispatch-zeroclaw --test smoke -- --ignored --nocapture
//! ```
//!
//! `ZEROCLAW_SMOKE_TOKEN` may be omitted against a gateway run without
//! pairing. Without the URL the tests skip rather than fail, so an unguarded
//! `--ignored` run in CI stays green.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pipecrab_core::{DispatchCommand, DispatchEvent};
use pipecrab_dispatch::{DispatchSink, DispatchSource};
use pipecrab_dispatch_zeroclaw::{ZeroclawConfig, ZeroclawSource, connect};
use url::Url;

/// A tool call id unique to this run.
///
/// The adapter sends the tool call id as `X-Idempotency-Key`, and the gateway
/// remembers keys it has already answered — a fixed id would make every run
/// after the first a `duplicate`, and so a `Failure` rather than the
/// `Completion` these tests wait for.
fn unique_id(label: &str) -> Arc<str> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the clock is past the epoch")
        .as_nanos();
    Arc::from(format!("{label}-{nanos}"))
}

/// The gateway URL (and optional token) from the environment, or `None` to skip.
fn smoke_config() -> Option<ZeroclawConfig> {
    let url = std::env::var("ZEROCLAW_SMOKE_URL").ok()?;
    let mut config = ZeroclawConfig::new()
        .with_base_url(Url::parse(&url).expect("ZEROCLAW_SMOKE_URL must be a valid URL"))
        .with_request_timeout(Duration::from_secs(120));
    if let Ok(token) = std::env::var("ZEROCLAW_SMOKE_TOKEN") {
        config = config.with_token(token);
    }
    Some(config)
}

/// Await events until one satisfies `pred`, printing the trail as it goes.
async fn wait_for(
    source: &mut ZeroclawSource,
    budget: Duration,
    pred: impl Fn(&DispatchEvent) -> bool,
) -> DispatchEvent {
    let deadline = tokio::time::Instant::now() + budget;
    loop {
        let event = tokio::time::timeout_at(deadline, source.next_event())
            .await
            .expect("an event before the deadline")
            .expect("the source must not error")
            .expect("the source stayed open");
        println!("  event: {event:?}");
        if pred(&event) {
            return event;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a live zeroclaw gateway; set ZEROCLAW_SMOKE_URL (and ZEROCLAW_SMOKE_TOKEN)"]
async fn live_round_trip_then_follow_up_in_the_same_session() {
    let Some(config) = smoke_config() else {
        eprintln!("skipping: set ZEROCLAW_SMOKE_URL to run this test");
        return;
    };
    let (mut source, sink) = connect(config);

    // A create whose answer proves the message reached the model.
    sink.send_command(DispatchCommand::Create {
        tool_call_id: unique_id("smoke-create"),
        task: Arc::from("Reply with exactly the word: pomegranate"),
        context: None,
    })
    .await
    .expect("create is accepted");

    let accepted = wait_for(&mut source, Duration::from_secs(10), |e| {
        matches!(e, DispatchEvent::Accepted { .. })
    })
    .await;
    let DispatchEvent::Accepted { task_id, .. } = accepted else {
        unreachable!()
    };

    let completion = wait_for(&mut source, Duration::from_secs(120), |e| {
        matches!(e, DispatchEvent::Completion { .. })
    })
    .await;
    let DispatchEvent::Completion { message, .. } = &completion else {
        unreachable!()
    };
    assert!(
        message.to_lowercase().contains("pomegranate"),
        "the reply carries the requested word: {message}"
    );

    // A follow-up under the same task: the replayed transcript must carry the
    // errand, or the agent cannot answer this.
    sink.send_command(DispatchCommand::Update {
        tool_call_id: unique_id("smoke-update"),
        task_id: task_id.clone(),
        message: Arc::from("What word did I just ask you to reply with?"),
    })
    .await
    .expect("update is accepted");

    let follow_up = wait_for(&mut source, Duration::from_secs(120), |e| {
        matches!(e, DispatchEvent::Completion { .. })
    })
    .await;
    let DispatchEvent::Completion { message, .. } = &follow_up else {
        unreachable!()
    };
    assert!(
        message.to_lowercase().contains("pomegranate"),
        "the session carried the conversation: {message}"
    );

    source.cancel();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a live zeroclaw gateway; set ZEROCLAW_SMOKE_URL (and ZEROCLAW_SMOKE_TOKEN)"]
async fn live_bad_token_fails_the_task_finally() {
    let Some(config) = smoke_config() else {
        eprintln!("skipping: set ZEROCLAW_SMOKE_URL to run this test");
        return;
    };
    // Only meaningful against a gateway that actually requires pairing.
    let (mut source, sink) = connect(config.with_token("definitely-not-a-valid-token"));

    sink.send_command(DispatchCommand::Create {
        tool_call_id: unique_id("smoke-bad-token"),
        task: Arc::from("this should never reach a model"),
        context: None,
    })
    .await
    .expect("create is accepted optimistically");

    let failure = wait_for(&mut source, Duration::from_secs(30), |e| {
        matches!(e, DispatchEvent::Failure { .. })
    })
    .await;
    let DispatchEvent::Failure { retryable, .. } = failure else {
        unreachable!()
    };
    assert!(!retryable, "an auth failure must not invite a retry");

    source.cancel();
}
