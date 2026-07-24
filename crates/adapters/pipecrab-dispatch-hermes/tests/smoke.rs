//! Live smoke test against a real `hermes gateway`. Ignored by default — it
//! needs a reachable gateway and burns real model time.
//!
//! ```sh
//! HERMES_SMOKE_URL=http://127.0.0.1:8642 \
//! HERMES_SMOKE_KEY=$API_SERVER_KEY \
//!   cargo test -p pipecrab-dispatch-hermes --test smoke -- --ignored --nocapture
//! ```
//!
//! Without both variables the tests skip rather than fail, so an unguarded
//! `--ignored` run in CI stays green.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use pipecrab_core::{DispatchCommand, DispatchEvent};
use pipecrab_dispatch::{DispatchSink, DispatchSource};
use pipecrab_dispatch_hermes::{HermesConfig, HermesSource, connect};
use url::Url;

/// The gateway URL and key from the environment, or `None` to skip.
fn smoke_config() -> Option<HermesConfig> {
    let url = std::env::var("HERMES_SMOKE_URL").ok()?;
    let key = std::env::var("HERMES_SMOKE_KEY").ok()?;
    Some(
        HermesConfig::new(key)
            .with_base_url(Url::parse(&url).expect("HERMES_SMOKE_URL must be a valid URL"))
            .with_poll_interval(Duration::from_secs(1))
            .with_request_timeout(Duration::from_secs(30)),
    )
}

/// Await events until one satisfies `pred`, printing the trail as it goes.
async fn wait_for(
    source: &mut HermesSource,
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
#[ignore = "requires a live hermes gateway; set HERMES_SMOKE_URL and HERMES_SMOKE_KEY"]
async fn live_gateway_round_trip() {
    let Some(config) = smoke_config() else {
        eprintln!("skipping: set HERMES_SMOKE_URL and HERMES_SMOKE_KEY to run this test");
        return;
    };

    let (mut source, sink) = connect(config);

    // 1. dispatch_task → Accepted.
    sink.send_command(DispatchCommand::Create {
        tool_call_id: Arc::from("smoke-call-1"),
        task: Arc::from("Reply with exactly the word: pipecrab"),
        context: None,
    })
    .await
    .expect("create is accepted");

    let accepted = wait_for(&mut source, Duration::from_secs(30), |e| {
        matches!(e, DispatchEvent::Accepted { .. })
    })
    .await;
    let task_id = match accepted {
        DispatchEvent::Accepted { task_id, .. } => task_id,
        _ => unreachable!(),
    };
    println!("accepted task_id = {task_id}");

    // 2. The run completes (via Progress updates) within a generous budget.
    let completion = wait_for(&mut source, Duration::from_secs(180), |e| {
        matches!(
            e,
            DispatchEvent::Completion { .. } | DispatchEvent::Failure { .. }
        )
    })
    .await;
    let DispatchEvent::Completion { message, .. } = &completion else {
        panic!("the run did not complete: {completion:?}");
    };
    assert!(!message.is_empty(), "a completion carries its output");

    // 3. update_task chains a follow-up run under the same task id.
    sink.send_command(DispatchCommand::Update {
        tool_call_id: Arc::from("smoke-call-2"),
        task_id: task_id.clone(),
        message: Arc::from("Reply with exactly the word: again"),
    })
    .await
    .expect("update is accepted");

    let second = wait_for(&mut source, Duration::from_secs(180), |e| {
        matches!(
            e,
            DispatchEvent::Completion { .. }
                | DispatchEvent::Failure { .. }
                | DispatchEvent::Rejected { .. }
        )
    })
    .await;
    match &second {
        DispatchEvent::Completion { task_id: t, .. } => {
            assert_eq!(t, &task_id, "the follow-up run reports under the same task");
        }
        other => panic!("the chained run did not complete: {other:?}"),
    }

    // 4. cancel closes the source without touching the remote run.
    source.cancel();
    let closed = tokio::time::timeout(Duration::from_secs(5), source.next_event())
        .await
        .expect("closes promptly")
        .expect("no error on close");
    assert!(closed.is_none(), "cancel closes the source");
}
