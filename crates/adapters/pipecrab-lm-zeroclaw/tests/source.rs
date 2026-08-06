//! `ZeroclawDelegateSource` over a temp `delegate_results` directory — no
//! daemon required. Scans are driven deterministically through the
//! turn-settled notifier; the timed cadence only matters for staleness.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use pipecrab_core::DispatchEvent;
use pipecrab_dispatch::DispatchSource;
use pipecrab_lm_zeroclaw::{PollConfig, ZeroclawDelegateSource};
use serde_json::json;
use tokio::sync::Notify;

const WAIT: Duration = Duration::from_secs(5);
const QUIET: Duration = Duration::from_millis(250);

fn poll_config() -> PollConfig {
    PollConfig {
        interval: Duration::from_millis(25),
        settle_backoff: Duration::from_millis(50),
        stale_after: Duration::from_millis(400),
    }
}

/// Write a result file the way the daemon does: temp file, then rename.
fn write_result(
    dir: &Path,
    task_id: &str,
    status: &str,
    output: Option<&str>,
    error: Option<&str>,
    started_at: &str,
) {
    let body = json!({
        "task_id": task_id,
        "agent": "research",
        "status": status,
        "output": output,
        "error": error,
        "started_at": started_at,
        "finished_at": null,
    })
    .to_string();
    let tmp = dir.join(format!("{task_id}.json.tmp"));
    std::fs::write(&tmp, body).expect("write temp result");
    std::fs::rename(&tmp, dir.join(format!("{task_id}.json"))).expect("rename result");
}

/// The same formatting the daemon uses: RFC 3339 with a UTC offset and
/// sub-second precision, so "started after connect" holds even when the file
/// lands in the same second the source was constructed.
fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339()
}

struct Fixture {
    source: ZeroclawDelegateSource,
    notify: Arc<Notify>,
    dir: std::path::PathBuf,
    _tmp: tempfile::TempDir,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("delegate_results");
    std::fs::create_dir_all(&dir).expect("results dir");
    let notify = Arc::new(Notify::new());
    let source = ZeroclawDelegateSource::watch(&dir, poll_config(), Arc::clone(&notify));
    Fixture {
        source,
        notify,
        dir,
        _tmp: tmp,
    }
}

async fn next_event(source: &mut ZeroclawDelegateSource) -> DispatchEvent {
    tokio::time::timeout(WAIT, source.next_event())
        .await
        .expect("timed out waiting for a dispatch event")
        .expect("dispatch source errored")
        .expect("dispatch source closed")
}

async fn assert_quiet(source: &mut ZeroclawDelegateSource) {
    let outcome = tokio::time::timeout(QUIET, source.next_event()).await;
    assert!(
        outcome.is_err(),
        "expected no dispatch event, got {:?}",
        outcome.unwrap()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn completion_emits_once_with_task_id_and_agent_in_the_message() {
    let mut f = fixture();

    write_result(&f.dir, "task-1", "running", None, None, &now_rfc3339());
    f.notify.notify_waiters();
    // Pending now; the cadence keeps scanning while we flip it to terminal.
    write_result(
        &f.dir,
        "task-1",
        "completed",
        Some("the answer is 42"),
        None,
        &now_rfc3339(),
    );
    f.notify.notify_waiters();

    let event = next_event(&mut f.source).await;
    let DispatchEvent::Completion { task_id, message } = event else {
        panic!("expected Completion, got {event:?}");
    };
    assert_eq!(&*task_id, "task-1");
    assert!(message.contains("task task-1"), "message: {message}");
    assert!(message.contains("agent research"), "message: {message}");
    assert!(message.contains("the answer is 42"), "message: {message}");

    // Rewriting the terminal file must not re-emit.
    write_result(
        &f.dir,
        "task-1",
        "completed",
        Some("the answer is 42"),
        None,
        &now_rfc3339(),
    );
    f.notify.notify_waiters();
    assert_quiet(&mut f.source).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn failure_carries_the_error_and_is_not_retryable() {
    let mut f = fixture();

    write_result(
        &f.dir,
        "task-2",
        "failed",
        None,
        Some("child agent exploded"),
        &now_rfc3339(),
    );
    f.notify.notify_waiters();

    let event = next_event(&mut f.source).await;
    let DispatchEvent::Failure {
        task_id,
        message,
        retryable,
    } = event
    else {
        panic!("expected Failure, got {event:?}");
    };
    assert_eq!(&*task_id, "task-2");
    assert!(!retryable);
    assert!(
        message.contains("child agent exploded"),
        "message: {message}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelled_tasks_are_silent() {
    let mut f = fixture();

    write_result(&f.dir, "task-3", "cancelled", None, None, &now_rfc3339());
    f.notify.notify_waiters();
    assert_quiet(&mut f.source).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn files_from_before_this_session_are_ignored() {
    let mut f = fixture();

    write_result(
        &f.dir,
        "task-old",
        "completed",
        Some("stale from last week"),
        None,
        "2020-01-01T00:00:00+00:00",
    );
    f.notify.notify_waiters();
    assert_quiet(&mut f.source).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn unparseable_files_are_retried_until_valid() {
    let mut f = fixture();

    std::fs::write(f.dir.join("task-4.json"), "{ not json").expect("write garbage");
    f.notify.notify_waiters();
    assert_quiet(&mut f.source).await;

    write_result(
        &f.dir,
        "task-4",
        "completed",
        Some("eventually fine"),
        None,
        &now_rfc3339(),
    );
    f.notify.notify_waiters();
    let event = next_event(&mut f.source).await;
    assert!(
        matches!(event, DispatchEvent::Completion { ref task_id, .. } if &**task_id == "task-4"),
        "got {event:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn stuck_running_tasks_go_stale_and_later_terminals_are_ignored() {
    let mut f = fixture();

    write_result(&f.dir, "task-5", "running", None, None, &now_rfc3339());
    f.notify.notify_waiters();

    // The pending cadence rescans until stale_after (400 ms) elapses.
    let event = next_event(&mut f.source).await;
    let DispatchEvent::Failure {
        task_id,
        message,
        retryable,
    } = event
    else {
        panic!("expected staleness Failure, got {event:?}");
    };
    assert_eq!(&*task_id, "task-5");
    assert!(!retryable);
    assert!(message.contains("no terminal status"), "message: {message}");

    // A genuine terminal arriving after the staleness verdict is ignored.
    write_result(
        &f.dir,
        "task-5",
        "completed",
        Some("too late"),
        None,
        &now_rfc3339(),
    );
    f.notify.notify_waiters();
    assert_quiet(&mut f.source).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_closes_the_source_gracefully() {
    let mut f = fixture();

    f.source.cancel();
    let closed = tokio::time::timeout(WAIT, f.source.next_event())
        .await
        .expect("timed out waiting for close")
        .expect("close must be graceful");
    assert!(closed.is_none(), "expected Ok(None), got {closed:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_missing_results_directory_is_not_an_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("never-created");
    let notify = Arc::new(Notify::new());
    let mut source = ZeroclawDelegateSource::watch(&dir, poll_config(), Arc::clone(&notify));

    notify.notify_waiters();
    assert_quiet(&mut source).await;
}
