//! [`Inner`]: the shared heart both halves of the transport hold behind an
//! `Arc`. It owns the reqwest client, the task registry, the event channel
//! sender, and the cancellation state, and is the single place every event is
//! emitted from — so a task's trail can never drift from what actually reached
//! the pipeline.
//!
//! There is no poller: `POST /webhook` is synchronous, so each accepted
//! command spawns one detached worker ([`deliver`]) that performs the request
//! and emits its terminal event when the reply lands. Workers must be
//! detached — a whole agent turn can take minutes, and the egress stage's
//! `send_command` must not block for one.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use pipecrab_core::DispatchEvent;
use pipecrab_dispatch::DispatchError;
use tokio::sync::{Notify, mpsc};

use crate::classify::{Classified, classify};
use crate::client::{ZeroclawClient, compose_follow_up, compose_message};
use crate::task::{TaskEntry, TaskId, Turn};

/// Bounded capacity of the event channel. A worker *awaits* its send, so a
/// slow-draining pipeline backpressures the workers — natural, never loss.
const EVENT_CHANNEL_CAPACITY: usize = 64;

/// The shared transport state.
pub(crate) struct Inner {
    /// HTTP access to the gateway.
    client: ZeroclawClient,
    /// `TaskId` → its [`TaskEntry`]. A task lives here for the adapter's
    /// lifetime; webhook turns come and go under it.
    registry: Mutex<HashMap<TaskId, TaskEntry>>,
    /// The single persistent event sender. Taken (set to `None`) by
    /// [`cancel`](Self::cancel) so the channel can close once in-flight sends
    /// finish, which is how [`ZeroclawSource`](crate::ZeroclawSource) observes
    /// the graceful close. `emit` clones this per call and holds no lasting
    /// sender.
    events_tx: Mutex<Option<mpsc::Sender<DispatchEvent>>>,
    /// Set once by [`cancel`](Self::cancel); read by the sink (to reject
    /// sends) and the workers (to abandon in-flight requests).
    cancelled: AtomicBool,
    /// Wakes every worker parked on [`cancelled_wait`](Self::cancelled_wait)
    /// so an in-flight request is abandoned at once.
    cancel_wake: Notify,
}

impl Inner {
    /// Build the shared state plus the receive half of its event channel.
    pub(crate) fn new(client: ZeroclawClient) -> (Self, mpsc::Receiver<DispatchEvent>) {
        let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let inner = Self {
            client,
            registry: Mutex::new(HashMap::new()),
            events_tx: Mutex::new(Some(tx)),
            cancelled: AtomicBool::new(false),
            cancel_wake: Notify::new(),
        };
        (inner, rx)
    }

    // --- cancellation -----------------------------------------------------

    /// Whether the transport has been cancelled.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Flip the stop signal, drop the persistent event sender so the source's
    /// channel can close, and wake every parked worker. Idempotent,
    /// non-blocking, no HTTP; deliberately leaves the gateway to finish its
    /// turns.
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        *self.events_tx.lock().unwrap() = None;
        self.cancel_wake.notify_waiters();
    }

    /// Resolve once the transport is cancelled. Registers with the wake before
    /// checking the flag, so a `cancel` racing this call is never missed.
    async fn cancelled_wait(&self) {
        loop {
            let notified = self.cancel_wake.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }

    // --- emission ---------------------------------------------------------

    /// Emit one event: append it to its task's ring (when the task exists) and
    /// send it on the channel. The single choke point for *all* emission, used
    /// by both the sink and the workers.
    ///
    /// A closed channel is a fatal [`DispatchError`]: the sink surfaces it,
    /// and a worker treats it as its signal to stop.
    pub(crate) async fn emit(&self, event: DispatchEvent) -> Result<(), DispatchError> {
        // Record in the task's trail first. Scope the guard so it is dropped
        // before the await below — never hold the registry lock across a send.
        if let Some(task_id) = task_id_of(&event) {
            let mut registry = self.registry.lock().unwrap();
            if let Some(entry) = registry.get_mut(&task_id) {
                entry.events.push(SystemTime::now(), event.clone());
            }
        }
        // Clone the live sender, if any. A worker's terminal event may reach
        // here after `cancel` has taken the sender; that is still a closed
        // channel.
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

    /// Handle a `Create`: mint a task, emit `Accepted`, and spawn the worker
    /// that performs its first webhook turn. `Err` only on channel breakage.
    ///
    /// `Accepted` is optimistic — the gateway's answer takes a whole agent
    /// turn to arrive, so there is no cheap acceptance to await first. A
    /// gateway-level refusal surfaces as this task's `Failure`.
    pub(crate) async fn create(
        self: &Arc<Self>,
        tool_call_id: Arc<str>,
        task: Arc<str>,
        context: Option<Arc<str>>,
    ) -> Result<(), DispatchError> {
        let task_id = TaskId::mint();
        let message = compose_message(&task, context.as_deref());
        // Insert the entry *before* emitting so `Accepted` lands in its ring.
        self.registry
            .lock()
            .unwrap()
            .insert(task_id.clone(), TaskEntry::new());
        self.emit(DispatchEvent::Accepted {
            tool_call_id: tool_call_id.clone(),
            task_id: task_id.as_arc(),
        })
        .await?;
        // A create opens the conversation: nothing to replay, so the wire
        // message and the recorded user turn are the same text.
        tokio::spawn(deliver(
            Arc::clone(self),
            task_id,
            message.clone(),
            message,
            tool_call_id,
        ));
        Ok(())
    }

    /// Handle an `Update`: reject an unknown task or one whose turn is still
    /// executing, otherwise spawn the worker that posts the follow-up with the
    /// task's transcript replayed ahead of it.
    pub(crate) async fn update(
        self: &Arc<Self>,
        tool_call_id: Arc<str>,
        task_id: Arc<str>,
        message: Arc<str>,
    ) -> Result<(), DispatchError> {
        let task_id = TaskId::from(task_id);
        // Classify readiness, claim the in-flight slot, and read the
        // transcript under one lock, so two racing updates cannot both post
        // for this task or replay the same history.
        let readiness = {
            let mut registry = self.registry.lock().unwrap();
            match registry.get_mut(&task_id) {
                None => Readiness::Unknown,
                Some(entry) if entry.in_flight => Readiness::Running,
                Some(entry) => {
                    entry.in_flight = true;
                    Readiness::Idle(entry.transcript.snapshot())
                }
            }
        };

        match readiness {
            Readiness::Unknown => {
                self.emit(DispatchEvent::Rejected {
                    tool_call_id,
                    message: Arc::from("unknown task_id"),
                })
                .await
            }
            Readiness::Running => {
                self.emit(DispatchEvent::Rejected {
                    tool_call_id,
                    message: Arc::from(
                        "task is still running; the ZeroClaw webhook API has no way \
                         to deliver input to an executing turn",
                    ),
                })
                .await
            }
            Readiness::Idle(transcript) => {
                tokio::spawn(deliver(
                    Arc::clone(self),
                    task_id,
                    compose_follow_up(&transcript, &message),
                    message.to_string(),
                    tool_call_id,
                ));
                Ok(())
            }
        }
    }

    /// Clear a task's in-flight flag once its request resolves (or is
    /// abandoned by `cancel`), so a later update can chain a new turn.
    fn finish(&self, task_id: &TaskId) {
        if let Some(entry) = self.registry.lock().unwrap().get_mut(task_id) {
            entry.in_flight = false;
        }
    }

    /// Append one answered exchange to a task's transcript, for its next turn
    /// to replay.
    fn record_turn(&self, task_id: &TaskId, turn: Turn) {
        if let Some(entry) = self.registry.lock().unwrap().get_mut(task_id) {
            entry.transcript.push(turn);
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

/// One webhook turn, run detached: POST `wire`, classify the reply, record the
/// exchange, emit the terminal event. Racing a `cancel` abandons the request —
/// the gateway finishes the turn on its own, but there is no channel left to
/// tell.
///
/// `wire` is what the gateway receives (a follow-up's includes the replayed
/// transcript); `user` is the message as the user wrote it, and is what the
/// transcript keeps.
async fn deliver(
    inner: Arc<Inner>,
    task_id: TaskId,
    wire: String,
    user: String,
    idempotency_key: Arc<str>,
) {
    let outcome = tokio::select! {
        biased;
        () = inner.cancelled_wait() => None,
        outcome = inner
            .client
            .post_message(task_id.as_str(), &wire, &idempotency_key) => Some(outcome),
    };
    inner.finish(&task_id);
    let Some(outcome) = outcome else {
        return;
    };
    let event = match classify(&outcome) {
        Classified::Completed { message } => {
            // Only an answered turn joins the transcript: a failed one left no
            // reply, and replaying the question alone would misreport it as
            // something the agent saw.
            inner.record_turn(
                &task_id,
                Turn {
                    user,
                    agent: message.clone(),
                },
            );
            DispatchEvent::Completion {
                task_id: task_id.as_arc(),
                message: Arc::from(message),
            }
        }
        Classified::Failed { message, retryable } => DispatchEvent::Failure {
            task_id: task_id.as_arc(),
            message: Arc::from(message),
            retryable,
        },
    };
    // A closed channel just ends the worker; there is no one left to tell.
    let _ = inner.emit(event).await;
}

/// A task's readiness to accept an update.
enum Readiness {
    /// No such task.
    Unknown,
    /// A turn is executing; the webhook API cannot inject input into one.
    Running,
    /// Between turns; a follow-up can be posted, carrying this transcript.
    Idle(Vec<Turn>),
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
    use crate::config::ZeroclawConfig;

    /// An `Inner` whose HTTP client points nowhere (unused in these registry
    /// tests — every call here manipulates the registry directly).
    fn inner() -> Inner {
        Inner::new(ZeroclawClient::new(&ZeroclawConfig::new())).0
    }

    fn seed(inner: &Inner, task_id: &TaskId, in_flight: bool) {
        let mut entry = TaskEntry::new();
        entry.in_flight = in_flight;
        inner
            .registry
            .lock()
            .unwrap()
            .insert(task_id.clone(), entry);
    }

    #[test]
    fn finish_clears_in_flight_but_keeps_the_entry() {
        let inner = inner();
        let task = TaskId::from("t1");
        seed(&inner, &task, true);

        inner.finish(&task);

        let registry = inner.registry.lock().unwrap();
        let entry = registry.get(&task).expect("entry preserved");
        assert!(!entry.in_flight, "in-flight flag cleared");
    }

    #[test]
    fn finish_on_an_unknown_task_is_a_no_op() {
        let inner = inner();
        inner.finish(&TaskId::from("nope"));
        assert!(inner.registry.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn cancelled_wait_resolves_for_a_cancel_before_and_after() {
        let inner = Arc::new(inner());

        // Cancel first, wait after: resolves immediately.
        inner.cancel();
        inner.cancelled_wait().await;

        // Wait first, cancel after: the parked waiter is woken.
        let (fresh, _rx) = Inner::new(ZeroclawClient::new(&ZeroclawConfig::new()));
        let fresh = Arc::new(fresh);
        let waiter = tokio::spawn({
            let fresh = Arc::clone(&fresh);
            async move { fresh.cancelled_wait().await }
        });
        // Let the waiter park before cancelling.
        tokio::task::yield_now().await;
        fresh.cancel();
        waiter.await.expect("the waiter resolves");
    }

    #[test]
    fn rejected_has_no_task_and_is_skipped_by_the_ring() {
        assert!(
            task_id_of(&DispatchEvent::Rejected {
                tool_call_id: Arc::from("c"),
                message: Arc::from("no"),
            })
            .is_none()
        );
    }
}
