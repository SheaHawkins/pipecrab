//! [`Inner`]: the shared heart both halves of the transport hold behind an
//! `Arc`. It owns the reqwest client, the task registry, the event channel
//! sender, and the cancellation state, and is the single place every event is
//! emitted from — so a task's trail can never drift from what actually reached
//! the pipeline.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use pipecrab_core::DispatchEvent;
use pipecrab_dispatch::DispatchError;
use tokio::sync::{Notify, mpsc};

use crate::classify::{Classified, classify};
use crate::client::{HermesClient, PollResult, compose_input};
use crate::task::{HistoryTurn, RunId, TaskEntry, TaskId};

/// Bounded capacity of the event channel. The poller *awaits* its sends, so a
/// slow-draining pipeline fills this and pauses polling — natural backpressure,
/// never loss.
const EVENT_CHANNEL_CAPACITY: usize = 64;

/// The shared transport state.
pub(crate) struct Inner {
    /// HTTP access to the gateway.
    client: HermesClient,
    /// How often the poller sweeps active runs.
    poll_interval: Duration,
    /// `TaskId` → its [`TaskEntry`]. A task lives here for the adapter's
    /// lifetime; runs come and go under it.
    registry: Mutex<HashMap<TaskId, TaskEntry>>,
    /// The single persistent event sender. Taken (set to `None`) by
    /// [`cancel`](Self::cancel) so the channel can close once in-flight sends
    /// finish, which is how [`HermesSource`](crate::HermesSource) observes the
    /// graceful close. `emit` clones this per call and holds no lasting sender.
    events_tx: Mutex<Option<mpsc::Sender<DispatchEvent>>>,
    /// Set once by [`cancel`](Self::cancel); read by the sink (to reject sends)
    /// and the poller (to stop).
    cancelled: AtomicBool,
    /// Wakes the poller so it observes [`cancelled`](Self::cancelled) at once,
    /// and polls a freshly created/re-armed run without waiting a full interval.
    wake: Notify,
}

impl Inner {
    /// Build the shared state plus the receive half of its event channel.
    pub(crate) fn new(
        client: HermesClient,
        poll_interval: Duration,
    ) -> (Self, mpsc::Receiver<DispatchEvent>) {
        let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let inner = Self {
            client,
            poll_interval,
            registry: Mutex::new(HashMap::new()),
            events_tx: Mutex::new(Some(tx)),
            cancelled: AtomicBool::new(false),
            wake: Notify::new(),
        };
        (inner, rx)
    }

    /// The interval the poller sweeps on.
    pub(crate) fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    // --- cancellation -----------------------------------------------------

    /// Whether the transport has been cancelled.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Flip the stop signal, drop the persistent event sender so the source's
    /// channel can close, and wake the poller. Idempotent, non-blocking, no HTTP;
    /// deliberately does not touch remote runs.
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        *self.events_tx.lock().unwrap() = None;
        self.wake.notify_one();
    }

    /// Await the poller's wake signal (a `cancel`, a new run, or nothing yet).
    pub(crate) async fn woken(&self) {
        self.wake.notified().await;
    }

    // --- emission ---------------------------------------------------------

    /// Emit one event: append it to its task's ring (when the task exists) and
    /// send it on the channel. The single choke point for *all* emission, used
    /// by both the sink and the poller.
    ///
    /// A closed channel is a fatal [`DispatchError`]: the sink surfaces it, and
    /// the poller treats it as its signal to stop.
    pub(crate) async fn emit(&self, event: DispatchEvent) -> Result<(), DispatchError> {
        // Record in the task's trail first. Scope the guard so it is dropped
        // before the await below — never hold the registry lock across a send.
        if let Some(task_id) = task_id_of(&event) {
            let mut registry = self.registry.lock().unwrap();
            if let Some(entry) = registry.get_mut(&task_id) {
                entry.events.push(SystemTime::now(), event.clone());
            }
        }
        // Clone the live sender, if any. A create-time `Rejected` may reach here
        // after `cancel` has taken the sender; that is still a closed channel.
        let sender = self.events_tx.lock().unwrap().clone();
        match sender {
            Some(sender) => sender
                .send(event)
                .await
                .map_err(|_| DispatchError::fatal("dispatch event channel closed")),
            None => Err(DispatchError::fatal("dispatch event channel closed")),
        }
    }

    // --- commands (sink side) --------------------------------------------

    /// Handle a `Create`: mint a task, post a run, and emit `Accepted` on
    /// success or `Rejected` on any HTTP failure. Never returns a rejection as an
    /// `Err` — only channel breakage does.
    pub(crate) async fn create(
        &self,
        tool_call_id: std::sync::Arc<str>,
        task: std::sync::Arc<str>,
        context: Option<std::sync::Arc<str>>,
    ) -> Result<(), DispatchError> {
        let task_id = TaskId::mint();
        let input = compose_input(&task, context.as_deref());
        // A first run opens the conversation, so it replays no history.
        match self
            .client
            .create_run(&input, task_id.as_str(), &tool_call_id, &[])
            .await
        {
            Ok(run_id) => {
                // Insert the entry *before* emitting so `Accepted` lands in its
                // ring.
                self.registry
                    .lock()
                    .unwrap()
                    .insert(task_id.clone(), TaskEntry::new(run_id, input));
                let accepted = DispatchEvent::Accepted {
                    tool_call_id,
                    task_id: task_id.as_arc(),
                };
                self.emit(accepted).await?;
                // Poll the new run promptly rather than waiting a whole interval.
                self.wake.notify_one();
                Ok(())
            }
            Err(message) => {
                self.emit(DispatchEvent::Rejected {
                    tool_call_id,
                    message,
                })
                .await
            }
        }
    }

    /// Handle an `Update`: reject an unknown or still-running task, otherwise
    /// chain a new run into the same session. Success emits no event — the new
    /// run's state arrives via polling.
    pub(crate) async fn update(
        &self,
        tool_call_id: std::sync::Arc<str>,
        task_id: std::sync::Arc<str>,
        message: std::sync::Arc<str>,
    ) -> Result<(), DispatchError> {
        let task_id = TaskId::from(task_id);
        // Classify readiness and snapshot the history under the lock, then drop
        // it — the POST below must not hold it.
        let readiness = {
            let registry = self.registry.lock().unwrap();
            match registry.get(&task_id) {
                None => Readiness::Unknown,
                Some(entry) if entry.active_run.is_some() => Readiness::Running,
                Some(entry) => Readiness::Idle(entry.history.clone()),
            }
        };

        match readiness {
            Readiness::Unknown => {
                self.emit(DispatchEvent::Rejected {
                    tool_call_id,
                    message: std::sync::Arc::from("unknown task_id"),
                })
                .await
            }
            Readiness::Running => {
                self.emit(DispatchEvent::Rejected {
                    tool_call_id,
                    message: std::sync::Arc::from(
                        "task is still running; the Hermes runs API has no endpoint \
                         that delivers input to an in-flight run",
                    ),
                })
                .await
            }
            Readiness::Idle(history) => match self
                .client
                .create_run(&message, task_id.as_str(), &tool_call_id, &history)
                .await
            {
                Ok(run_id) => {
                    if let Some(entry) = self.registry.lock().unwrap().get_mut(&task_id) {
                        entry.active_run = Some(run_id);
                        // Re-arm progress dedupe for the new run.
                        entry.last_status = None;
                        entry.history.push(HistoryTurn::user(message.to_string()));
                    }
                    self.wake.notify_one();
                    Ok(())
                }
                Err(message) => {
                    self.emit(DispatchEvent::Rejected {
                        tool_call_id,
                        message,
                    })
                    .await
                }
            },
        }
    }

    // --- polling (poller side) -------------------------------------------

    /// One sweep of every active run. Returns `Err` only when the event channel
    /// has closed — the poller's signal to stop.
    pub(crate) async fn poll_once(&self) -> Result<(), DispatchError> {
        let active: Vec<(TaskId, RunId)> = {
            let registry = self.registry.lock().unwrap();
            registry
                .iter()
                .filter_map(|(task_id, entry)| {
                    entry
                        .active_run
                        .clone()
                        .map(|run_id| (task_id.clone(), run_id))
                })
                .collect()
        };

        for (task_id, run_id) in active {
            match self.client.get_run(&run_id).await {
                PollResult::Status(view) => {
                    let classified =
                        classify(&view.status, view.output.as_deref(), view.error.as_deref());
                    if let Some(event) = self.apply(&task_id, &run_id, classified) {
                        self.emit(event).await?;
                    }
                }
                PollResult::Gone => {
                    if self.clear_run(&task_id, &run_id) {
                        self.emit(DispatchEvent::Failure {
                            task_id: task_id.as_arc(),
                            message: std::sync::Arc::from(
                                "run state expired before it was observed",
                            ),
                            retryable: false,
                        })
                        .await?;
                    }
                }
                // The run is still real; log and retry next tick.
                PollResult::Transient(reason) => {
                    eprintln!("pipecrab-dispatch-hermes: {reason}");
                }
            }
        }
        Ok(())
    }

    /// Fold one classified observation into the registry and return the event to
    /// emit, if any. Guards the whole transition under the lock (no await
    /// inside): a terminal status clears the active run; a status change updates
    /// `last_status`; an unchanged non-terminal status yields nothing.
    fn apply(
        &self,
        task_id: &TaskId,
        run_id: &RunId,
        classified: Classified,
    ) -> Option<DispatchEvent> {
        let mut registry = self.registry.lock().unwrap();
        let entry = registry.get_mut(task_id)?;
        // Ignore an observation for a run this task has already moved past (a
        // late poll landing after an update re-armed the task with a new run).
        if entry.active_run.as_ref() != Some(run_id) {
            return None;
        }
        match classified {
            Classified::Completed { message } => {
                entry.active_run = None;
                // The answer becomes context for any follow-up run.
                entry.history.push(HistoryTurn::assistant(message.clone()));
                Some(DispatchEvent::Completion {
                    task_id: task_id.as_arc(),
                    message: std::sync::Arc::from(message),
                })
            }
            Classified::Failed { message } => {
                entry.active_run = None;
                Some(DispatchEvent::Failure {
                    task_id: task_id.as_arc(),
                    message: std::sync::Arc::from(message),
                    retryable: false,
                })
            }
            Classified::Cancelled => {
                entry.active_run = None;
                Some(DispatchEvent::Failure {
                    task_id: task_id.as_arc(),
                    message: std::sync::Arc::from("run cancelled"),
                    retryable: true,
                })
            }
            Classified::InFlight { status } => {
                if entry.last_status.as_deref() == Some(status.as_str()) {
                    return None;
                }
                entry.last_status = Some(status.clone());
                Some(DispatchEvent::Progress {
                    task_id: task_id.as_arc(),
                    message: std::sync::Arc::from(status),
                })
            }
        }
    }

    /// Clear a task's active run iff it is still `run_id`. Returns whether a
    /// clear happened — used to fire the expired-state `Failure` only once.
    fn clear_run(&self, task_id: &TaskId, run_id: &RunId) -> bool {
        let mut registry = self.registry.lock().unwrap();
        match registry.get_mut(task_id) {
            Some(entry) if entry.active_run.as_ref() == Some(run_id) => {
                entry.active_run = None;
                true
            }
            _ => false,
        }
    }

    // --- inspection (sink side) ------------------------------------------

    /// A cloned snapshot of a task's emitted-event trail, if the task exists.
    pub(crate) fn task_events(&self, task_id: &TaskId) -> Option<Vec<(SystemTime, DispatchEvent)>> {
        self.registry
            .lock()
            .unwrap()
            .get(task_id)
            .map(|entry| entry.events.snapshot())
    }
}

/// A task's readiness to accept an update.
enum Readiness {
    /// No such task.
    Unknown,
    /// A run is executing; the runs API cannot inject input into one.
    Running,
    /// Between runs; a new run can be chained, replaying this history.
    Idle(Vec<HistoryTurn>),
}

/// The `task_id` an event concerns, or `None` for a pre-task `Rejected`.
fn task_id_of(event: &DispatchEvent) -> Option<TaskId> {
    match event {
        DispatchEvent::Accepted { task_id, .. }
        | DispatchEvent::Progress { task_id, .. }
        | DispatchEvent::Question { task_id, .. }
        | DispatchEvent::Completion { task_id, .. }
        | DispatchEvent::Failure { task_id, .. } => Some(TaskId::from(task_id.clone())),
        DispatchEvent::Rejected { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::HermesClient;
    use crate::config::HermesConfig;

    /// An `Inner` whose HTTP client points nowhere (unused in these registry
    /// tests — every call here manipulates the registry directly).
    fn inner() -> Inner {
        let config = HermesConfig::new("test-key");
        Inner::new(HermesClient::new(&config), config.poll_interval).0
    }

    fn seed(inner: &Inner, task_id: &TaskId, run: &str) {
        inner.registry.lock().unwrap().insert(
            task_id.clone(),
            TaskEntry::new(RunId::from(run.to_owned()), "the original task"),
        );
    }

    #[test]
    fn in_flight_progress_dedupes_on_status_change() {
        let inner = inner();
        let task = TaskId::from("t1");
        let run = RunId::from("run-1".to_owned());
        seed(&inner, &task, "run-1");

        // First `running` observation emits Progress and records the status.
        let first = inner.apply(
            &task,
            &run,
            Classified::InFlight {
                status: "running".into(),
            },
        );
        assert!(matches!(first, Some(DispatchEvent::Progress { .. })));
        // The same status again emits nothing.
        let repeat = inner.apply(
            &task,
            &run,
            Classified::InFlight {
                status: "running".into(),
            },
        );
        assert!(repeat.is_none());
        // A changed status emits Progress again.
        let changed = inner.apply(
            &task,
            &run,
            Classified::InFlight {
                status: "stopping".into(),
            },
        );
        assert!(matches!(changed, Some(DispatchEvent::Progress { .. })));
    }

    #[test]
    fn terminal_status_clears_active_run_but_keeps_entry() {
        let inner = inner();
        let task = TaskId::from("t2");
        let run = RunId::from("run-2".to_owned());
        seed(&inner, &task, "run-2");

        let event = inner.apply(
            &task,
            &run,
            Classified::Completed {
                message: "done".into(),
            },
        );
        assert!(matches!(event, Some(DispatchEvent::Completion { .. })));

        let registry = inner.registry.lock().unwrap();
        let entry = registry
            .get(&task)
            .expect("entry preserved after completion");
        assert!(entry.active_run.is_none(), "active run cleared");
    }

    #[test]
    fn a_completion_records_the_answer_for_follow_up_runs() {
        let inner = inner();
        let task = TaskId::from("t-hist");
        let run = RunId::from("run-h".to_owned());
        seed(&inner, &task, "run-h");

        inner.apply(
            &task,
            &run,
            Classified::Completed {
                message: "the answer".into(),
            },
        );

        let registry = inner.registry.lock().unwrap();
        let history = &registry.get(&task).unwrap().history;
        assert_eq!(
            history,
            &vec![
                HistoryTurn::user("the original task"),
                HistoryTurn::assistant("the answer"),
            ],
            "a follow-up run replays the task and its answer"
        );
    }

    #[test]
    fn a_failure_records_no_assistant_turn() {
        let inner = inner();
        let task = TaskId::from("t-fail");
        let run = RunId::from("run-f".to_owned());
        seed(&inner, &task, "run-f");

        inner.apply(
            &task,
            &run,
            Classified::Failed {
                message: "boom".into(),
            },
        );

        let registry = inner.registry.lock().unwrap();
        assert_eq!(
            registry.get(&task).unwrap().history,
            vec![HistoryTurn::user("the original task")],
            "a run that produced no answer adds nothing to say"
        );
    }

    #[test]
    fn cancelled_is_retryable_failure() {
        let inner = inner();
        let task = TaskId::from("t3");
        let run = RunId::from("run-3".to_owned());
        seed(&inner, &task, "run-3");

        match inner.apply(&task, &run, Classified::Cancelled) {
            Some(DispatchEvent::Failure { retryable, .. }) => assert!(retryable),
            other => panic!("expected retryable Failure, got {other:?}"),
        }
    }

    #[test]
    fn stale_run_observation_is_ignored() {
        let inner = inner();
        let task = TaskId::from("t4");
        seed(&inner, &task, "run-current");
        // A completion for a run the task has moved past changes nothing.
        let stale = RunId::from("run-old".to_owned());
        let event = inner.apply(
            &task,
            &stale,
            Classified::Completed {
                message: "late".into(),
            },
        );
        assert!(event.is_none());
        assert!(
            inner
                .registry
                .lock()
                .unwrap()
                .get(&task)
                .unwrap()
                .active_run
                .is_some(),
            "the current run is untouched"
        );
    }

    #[test]
    fn clear_run_only_clears_the_named_run() {
        let inner = inner();
        let task = TaskId::from("t5");
        seed(&inner, &task, "run-5");
        assert!(!inner.clear_run(&task, &RunId::from("other".to_owned())));
        assert!(inner.clear_run(&task, &RunId::from("run-5".to_owned())));
        // Second clear is a no-op (already cleared).
        assert!(!inner.clear_run(&task, &RunId::from("run-5".to_owned())));
    }

    #[test]
    fn rejected_has_no_task_and_is_skipped_by_the_ring() {
        assert!(
            task_id_of(&DispatchEvent::Rejected {
                tool_call_id: std::sync::Arc::from("c"),
                message: std::sync::Arc::from("no"),
            })
            .is_none()
        );
    }
}
