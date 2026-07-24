//! The adapter's task-identity types and the per-task record it keeps.
//!
//! [`TaskId`] is the durable identity the model holds — minted by this adapter,
//! doubling as the Hermes `session_id`, and *not* the Hermes `run_id`. A task
//! outlives the runs executed on its behalf: [`RunId`]s come and go under a
//! single [`TaskEntry`], which lives for the adapter's lifetime.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;

use pipecrab_core::DispatchEvent;
use uuid::Uuid;

/// How many `(SystemTime, DispatchEvent)` pairs a task's [`EventRing`] retains.
const EVENT_RING_CAPACITY: usize = 32;

/// The durable identity of a dispatched task, minted by this adapter and sent to
/// Hermes as the run's `session_id`. Stable across the follow-up runs that
/// [`update_task`](pipecrab_core::DispatchCommand::Update) chains into the same
/// session, and handed back to the model as the `task_id` of
/// [`DispatchEvent::Accepted`].
///
/// Deliberately distinct from a [`RunId`]: one task, many runs.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TaskId(Arc<str>);

impl TaskId {
    /// Mint a fresh, unique task identity (`pc-{uuid}`).
    pub(crate) fn mint() -> Self {
        Self(Arc::from(format!("pc-{}", Uuid::new_v4().simple())))
    }

    /// The identity as a string slice — this is the Hermes `session_id`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The identity as the `Arc<str>` the native [`DispatchEvent`] frames carry.
    pub(crate) fn as_arc(&self) -> Arc<str> {
        self.0.clone()
    }
}

impl From<Arc<str>> for TaskId {
    fn from(id: Arc<str>) -> Self {
        Self(id)
    }
}

impl From<&str> for TaskId {
    fn from(id: &str) -> Self {
        Self(Arc::from(id))
    }
}

impl fmt::Debug for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskId({})", self.0)
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A Hermes `run_id` — the id of one agent turn executing on a task's behalf.
/// Server-assigned by `POST /v1/runs`; the poller `GET`s it. Internal: the model
/// never sees a run id, only its [`TaskId`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct RunId(Arc<str>);

impl RunId {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for RunId {
    fn from(id: String) -> Self {
        Self(Arc::from(id))
    }
}

/// A bounded, in-order trail of the events emitted for one task. Bounded because
/// a [`TaskEntry`] already lives for the adapter's lifetime; an unbounded log
/// would compound that. The oldest pair is dropped once the ring is full.
#[derive(Default)]
pub(crate) struct EventRing {
    events: VecDeque<(SystemTime, DispatchEvent)>,
}

impl EventRing {
    /// Append one emitted event, evicting the oldest if at capacity.
    pub(crate) fn push(&mut self, at: SystemTime, event: DispatchEvent) {
        if self.events.len() == EVENT_RING_CAPACITY {
            self.events.pop_front();
        }
        self.events.push_back((at, event));
    }

    /// A cloned, oldest-first snapshot of the trail.
    pub(crate) fn snapshot(&self) -> Vec<(SystemTime, DispatchEvent)> {
        self.events.iter().cloned().collect()
    }
}

/// One dispatched task as the adapter tracks it: the run currently executing on
/// its behalf (if any), the last status observed for that run (to dedupe
/// [`Progress`](DispatchEvent::Progress)), and the bounded trail of what has been
/// emitted for it.
///
/// The owning [`TaskId`] is the map key, so it is not duplicated here. Created by
/// a `Create`, re-armed by an `Update`; a terminal status clears
/// [`active_run`](Self::active_run) but keeps the entry, so a later update still
/// resolves.
pub(crate) struct TaskEntry {
    /// The run being polled, or `None` between runs (before the first, or after
    /// one reaches a terminal status).
    pub(crate) active_run: Option<RunId>,
    /// The last status seen for `active_run`; `None` re-arms `Progress` dedupe
    /// when a new run starts.
    pub(crate) last_status: Option<String>,
    /// The bounded trail of events emitted for this task.
    pub(crate) events: EventRing,
}

impl TaskEntry {
    /// A new entry whose first run is `run`.
    pub(crate) fn new(run: RunId) -> Self {
        Self {
            active_run: Some(run),
            last_status: None,
            events: EventRing::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(msg: &str) -> DispatchEvent {
        DispatchEvent::Progress {
            task_id: Arc::from("t"),
            message: Arc::from(msg),
        }
    }

    #[test]
    fn minted_ids_are_unique_and_prefixed() {
        let a = TaskId::mint();
        let b = TaskId::mint();
        assert_ne!(a, b);
        assert!(a.as_str().starts_with("pc-"));
    }

    #[test]
    fn ring_is_bounded_and_keeps_the_newest() {
        let mut ring = EventRing::default();
        for i in 0..(EVENT_RING_CAPACITY + 5) {
            ring.push(SystemTime::UNIX_EPOCH, progress(&i.to_string()));
        }
        let snap = ring.snapshot();
        assert_eq!(snap.len(), EVENT_RING_CAPACITY);
        // Oldest five (0..=4) were evicted; the trail now starts at 5.
        assert_eq!(snap.first().unwrap().1, progress("5"));
        assert_eq!(
            snap.last().unwrap().1,
            progress(&(EVENT_RING_CAPACITY + 4).to_string())
        );
    }
}
