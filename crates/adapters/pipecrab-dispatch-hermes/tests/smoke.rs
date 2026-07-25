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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a live hermes gateway; set HERMES_SMOKE_URL and HERMES_SMOKE_KEY"]
async fn live_update_task_carries_the_original_errand() {
    let Some(config) = smoke_config() else {
        eprintln!("skipping: set HERMES_SMOKE_URL and HERMES_SMOKE_KEY to run this test");
        return;
    };

    let (mut source, sink) = connect(config);
    sink.send_command(DispatchCommand::Create {
        tool_call_id: Arc::from("smoke-ctx-1"),
        task: Arc::from("Remember the code word ZANZIBAR. Acknowledge briefly."),
        context: None,
    })
    .await
    .expect("create is accepted");

    let accepted = wait_for(&mut source, Duration::from_secs(60), |e| {
        matches!(e, DispatchEvent::Accepted { .. })
    })
    .await;
    let DispatchEvent::Accepted { task_id, .. } = accepted else {
        unreachable!()
    };
    wait_for(&mut source, Duration::from_secs(180), |e| {
        matches!(e, DispatchEvent::Completion { .. })
    })
    .await;

    sink.send_command(DispatchCommand::Update {
        tool_call_id: Arc::from("smoke-ctx-2"),
        task_id: task_id.clone(),
        message: Arc::from("What code word did I give you? Answer with just the word."),
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
        DispatchEvent::Completion { message, .. } => assert!(
            message.to_uppercase().contains("ZANZIBAR"),
            "the follow-up run must see the original errand, got {message:?}"
        ),
        other => panic!("the follow-up did not complete: {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a live hermes gateway; set HERMES_SMOKE_URL and HERMES_SMOKE_KEY"]
async fn live_concurrent_tasks_are_polled_independently() {
    let Some(config) = smoke_config() else {
        eprintln!("skipping: set HERMES_SMOKE_URL and HERMES_SMOKE_KEY to run this test");
        return;
    };

    // Three runs in flight at once: the poller sweeps them all each tick and
    // must keep their events apart.
    let words = ["alpha", "beta", "gamma"];
    let (mut source, sink) = connect(config);
    for (i, word) in words.iter().enumerate() {
        sink.send_command(DispatchCommand::Create {
            tool_call_id: Arc::from(format!("smoke-concurrent-{i}")),
            task: Arc::from(format!("Reply with exactly the word: {word}")),
            context: None,
        })
        .await
        .expect("create is accepted");
    }

    let mut expected: HashMap<Arc<str>, &str> = HashMap::new();
    for _ in 0..words.len() {
        let accepted = wait_for(&mut source, Duration::from_secs(60), |e| {
            matches!(e, DispatchEvent::Accepted { .. })
        })
        .await;
        let DispatchEvent::Accepted {
            tool_call_id,
            task_id,
        } = accepted
        else {
            unreachable!()
        };
        let i: usize = tool_call_id
            .rsplit('-')
            .next()
            .and_then(|n| n.parse().ok())
            .expect("the tool call id ends in its index");
        assert!(
            expected.insert(task_id, words[i]).is_none(),
            "each task gets a distinct id"
        );
    }
    assert_eq!(expected.len(), words.len());

    let deadline = tokio::time::Instant::now() + Duration::from_secs(240);
    let mut done: HashSet<Arc<str>> = HashSet::new();
    while done.len() < words.len() {
        let event = tokio::time::timeout_at(deadline, source.next_event())
            .await
            .expect("all tasks complete before the deadline")
            .expect("the source must not error")
            .expect("the source stayed open");
        println!("  event: {event:?}");
        match event {
            DispatchEvent::Completion { task_id, message } => {
                let want = expected.get(&task_id).expect("a task we dispatched");
                assert!(
                    message.to_lowercase().contains(want),
                    "task {task_id} answered {message:?}, expected {want:?}"
                );
                done.insert(task_id);
            }
            DispatchEvent::Failure {
                task_id, message, ..
            } => {
                panic!("task {task_id} failed: {message}")
            }
            _ => {}
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a live hermes gateway; set HERMES_SMOKE_URL and HERMES_SMOKE_KEY"]
async fn live_bad_api_key_is_rejected_not_an_error() {
    let Some(mut config) = smoke_config() else {
        eprintln!("skipping: set HERMES_SMOKE_URL and HERMES_SMOKE_KEY to run this test");
        return;
    };
    config.api_key = Arc::from("definitely-not-a-valid-api-key");

    let (mut source, sink) = connect(config);
    sink.send_command(DispatchCommand::Create {
        tool_call_id: Arc::from("smoke-badkey"),
        task: Arc::from("this should never run"),
        context: None,
    })
    .await
    .expect("a rejected create is still Ok at the sink");

    let rejected = wait_for(&mut source, Duration::from_secs(30), |e| {
        matches!(e, DispatchEvent::Rejected { .. })
    })
    .await;
    let DispatchEvent::Rejected {
        tool_call_id,
        message,
    } = rejected
    else {
        unreachable!()
    };
    assert_eq!(&*tool_call_id, "smoke-badkey");
    assert!(
        message.contains("401"),
        "message names the status: {message}"
    );
    assert!(
        !message.contains("definitely-not-a-valid-api-key"),
        "the key must never reach an error message: {message}"
    );
}
