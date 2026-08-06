//! [`ZeroclawDelegateSource`]: the delegation re-entry path.
//!
//! Background delegations (`delegate` with `background: true`) write
//! `{workspace_dir}/delegate_results/{task_id}.json` atomically — a `running`
//! record before the task spawns, a terminal record when it settles. The
//! poller scans the directory rather than tracking ids from tool-call
//! events, so tasks whose spawning turn was cancelled before its tool-result
//! update was observed are still caught.
//!
//! Scheduling: the poller sleeps until a turn settles, scans, then keeps
//! polling only while `running` tasks are pending — idle cost is zero.
//! There is no reaper for the result files (a killed daemon, a panicked
//! task, or a failed terminal write leaves `running` forever), so any task
//! still non-terminal `stale_after` after first seen is reported as a
//! failure and dropped.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use pipecrab_core::DispatchEvent;
use pipecrab_dispatch::{DispatchError, DispatchSource};
use serde::Deserialize;
use tokio::sync::{Notify, mpsc};
use tokio::time::Instant;

use crate::config::PollConfig;

/// How long a task set may stay pending before the scan cadence relaxes
/// from `interval` to `settle_backoff`.
const SETTLE_AFTER: Duration = Duration::from_secs(30);

/// Stop signal shared between the source handle and the poller task.
#[derive(Debug, Default)]
pub(crate) struct PollCancel {
    flag: AtomicBool,
    notify: Notify,
}

impl PollCancel {
    pub(crate) fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

/// The receive half of the delegation loop: emits a
/// [`DispatchEvent::Completion`] or [`DispatchEvent::Failure`] when a
/// background delegation settles. Owned single-threaded by
/// [`DispatchIngress`](pipecrab_dispatch::DispatchIngress) — `Send`, not
/// `Sync`, exactly as the trait allows.
pub struct ZeroclawDelegateSource {
    events: mpsc::Receiver<DispatchEvent>,
    cancel: Arc<PollCancel>,
}

impl ZeroclawDelegateSource {
    pub(crate) fn new(events: mpsc::Receiver<DispatchEvent>, cancel: Arc<PollCancel>) -> Self {
        Self { events, cancel }
    }

    /// Watch a `delegate_results` directory standalone, scanning whenever
    /// `turn_settled` is notified (and on the poll cadence while tasks are
    /// pending). [`connect`](crate::connect) wires this up automatically;
    /// this constructor exists for custom wiring and tests.
    ///
    /// Spawns the poller onto the ambient tokio runtime, which must outlive
    /// the source.
    pub fn watch(
        results_dir: impl Into<PathBuf>,
        poll: PollConfig,
        turn_settled: Arc<Notify>,
    ) -> Self {
        let (events_tx, events_rx) = mpsc::channel(64);
        let cancel = Arc::new(PollCancel::default());
        tokio::spawn(run_poller(
            results_dir.into(),
            poll,
            Utc::now(),
            turn_settled,
            Arc::clone(&cancel),
            events_tx,
        ));
        Self::new(events_rx, cancel)
    }
}

#[async_trait]
impl DispatchSource for ZeroclawDelegateSource {
    async fn next_event(&mut self) -> Result<Option<DispatchEvent>, DispatchError> {
        // `recv` is cancellation-safe; `None` (the poller wound down after
        // `cancel`, or the worker runtime is gone) is a graceful close.
        Ok(self.events.recv().await)
    }

    fn cancel(&self) {
        self.cancel.cancel();
    }
}

/// One `delegate_results/{task_id}.json` file, ZeroClaw's
/// `BackgroundDelegateResult` shape.
#[derive(Debug, Deserialize)]
struct DelegateResultFile {
    task_id: String,
    agent: String,
    status: DelegateStatus,
    #[serde(default)]
    output: Option<String>,
    #[serde(default)]
    error: Option<String>,
    started_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DelegateStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

pub(crate) async fn run_poller(
    results_dir: PathBuf,
    poll: PollConfig,
    connect_time: DateTime<Utc>,
    turn_settled: Arc<Notify>,
    cancel: Arc<PollCancel>,
    events: mpsc::Sender<DispatchEvent>,
) {
    // Terminal-emitted (or permanently ignored) task ids, keyed by file stem.
    let mut settled: HashSet<String> = HashSet::new();
    // First sighting of each not-yet-terminal file, for staleness.
    let mut pending: HashMap<String, Instant> = HashMap::new();
    let mut pending_since: Option<Instant> = None;

    loop {
        if cancel.is_cancelled() {
            return;
        }

        if pending.is_empty() {
            // Nothing in flight: sleep until a turn settles. Zero idle cost.
            tokio::select! {
                () = turn_settled.notified() => {}
                () = cancel.notify.notified() => {}
            }
        } else {
            let cadence = match pending_since {
                Some(since) if since.elapsed() > SETTLE_AFTER => poll.settle_backoff,
                _ => poll.interval,
            };
            tokio::select! {
                () = tokio::time::sleep(cadence) => {}
                () = turn_settled.notified() => {}
                () = cancel.notify.notified() => {}
            }
        }
        if cancel.is_cancelled() {
            return;
        }

        if !scan(
            &results_dir,
            &poll,
            connect_time,
            &mut settled,
            &mut pending,
            &events,
        )
        .await
        {
            return; // the event receiver is gone
        }

        pending_since = match (pending.is_empty(), pending_since) {
            (true, _) => None,
            (false, None) => Some(Instant::now()),
            (false, since) => since,
        };
    }
}

/// One directory scan. Returns `false` when the event channel closed.
async fn scan(
    results_dir: &std::path::Path,
    poll: &PollConfig,
    connect_time: DateTime<Utc>,
    settled: &mut HashSet<String>,
    pending: &mut HashMap<String, Instant>,
    events: &mpsc::Sender<DispatchEvent>,
) -> bool {
    // The daemon creates the directory on first delegation; absence means
    // no tasks yet, not an error.
    let Ok(entries) = std::fs::read_dir(results_dir) else {
        return true;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()).map(str::to_owned) else {
            continue;
        };
        if settled.contains(&stem) {
            continue;
        }

        let first_seen = *pending.entry(stem.clone()).or_insert_with(Instant::now);

        let parsed = std::fs::read_to_string(&path)
            .ok()
            .and_then(|body| serde_json::from_str::<DelegateResultFile>(&body).ok());
        let Some(result) = parsed else {
            // Unreadable or mid-drift: retry next scan; staleness backstops a
            // file that never becomes parseable.
            if first_seen.elapsed() > poll.stale_after
                && !emit_stale(&stem, settled, pending, events).await
            {
                return false;
            }
            continue;
        };

        // Files from before this session are someone else's tasks. An
        // unparseable timestamp is treated the same — provenance unknown.
        let started_in_session = DateTime::parse_from_rfc3339(&result.started_at)
            .map(|started| started.with_timezone(&Utc) >= connect_time)
            .unwrap_or(false);
        if !started_in_session {
            settled.insert(stem.clone());
            pending.remove(&stem);
            continue;
        }

        match result.status {
            DelegateStatus::Running => {
                if first_seen.elapsed() > poll.stale_after
                    && !emit_stale(&stem, settled, pending, events).await
                {
                    return false;
                }
            }
            DelegateStatus::Completed => {
                let message = format!(
                    "task {} (agent {}): {}",
                    result.task_id,
                    result.agent,
                    result.output.as_deref().unwrap_or("(no output)"),
                );
                settled.insert(stem.clone());
                pending.remove(&stem);
                let event = DispatchEvent::Completion {
                    task_id: result.task_id.into(),
                    message: message.into(),
                };
                if events.send(event).await.is_err() {
                    return false;
                }
            }
            DelegateStatus::Failed => {
                let message = format!(
                    "task {} (agent {}) failed: {}",
                    result.task_id,
                    result.agent,
                    result
                        .error
                        .as_deref()
                        .or(result.output.as_deref())
                        .unwrap_or("(no detail)"),
                );
                settled.insert(stem.clone());
                pending.remove(&stem);
                let event = DispatchEvent::Failure {
                    task_id: result.task_id.into(),
                    message: message.into(),
                    retryable: false,
                };
                if events.send(event).await.is_err() {
                    return false;
                }
            }
            DelegateStatus::Cancelled => {
                // The agent cancelled it itself via cancel_task; announcing
                // it would be noise.
                settled.insert(stem.clone());
                pending.remove(&stem);
            }
        }
    }
    true
}

/// Report a task that produced no terminal status within `stale_after`.
async fn emit_stale(
    stem: &str,
    settled: &mut HashSet<String>,
    pending: &mut HashMap<String, Instant>,
    events: &mpsc::Sender<DispatchEvent>,
) -> bool {
    settled.insert(stem.to_owned());
    pending.remove(stem);
    let event = DispatchEvent::Failure {
        task_id: Arc::from(stem),
        message: format!(
            "task {stem} produced no terminal status within the staleness window; \
             treating it as lost (the daemon may have restarted mid-task)"
        )
        .into(),
        retryable: false,
    };
    events.send(event).await.is_ok()
}
