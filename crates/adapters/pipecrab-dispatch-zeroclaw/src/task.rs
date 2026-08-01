//! The adapter's task-identity type and the per-task record it keeps.
//!
//! [`TaskId`] is the durable identity the model holds — minted by this adapter
//! and sent as the ZeroClaw `X-Session-Id`. A task outlives the webhook turns
//! executed on its behalf: a [`TaskEntry`] lives for the adapter's lifetime
//! while turns come and go under it, holding the [`Transcript`] that gives a
//! follow-up turn its conversation.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;

use pipecrab_core::DispatchEvent;
use uuid::Uuid;

/// How many `(SystemTime, DispatchEvent)` pairs a task's [`EventRing`] retains.
const EVENT_RING_CAPACITY: usize = 32;
/// How many completed exchanges a task's [`Transcript`] retains.
const TRANSCRIPT_CAPACITY: usize = 16;

/// The durable identity of a dispatched task, minted by this adapter and sent
/// to ZeroClaw as every request's `X-Session-Id`. Stable across the follow-up
/// turns that [`update_task`](pipecrab_core::DispatchCommand::Update) posts,
/// and handed back to the model as the `task_id` of
/// [`DispatchEvent::Accepted`].
///
/// The minted form (`pc-` plus a simple-format uuid) stays inside the
/// gateway's session-id constraints: ≤128 chars of `[A-Za-z0-9._-]`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TaskId(Arc<str>);

impl TaskId {
    /// Mint a fresh, unique task identity (`pc-{uuid}`).
    pub(crate) fn mint() -> Self {
        Self(Arc::from(format!("pc-{}", Uuid::new_v4().simple())))
    }

    /// The identity as a string slice — this is the ZeroClaw `X-Session-Id`.
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

/// A bounded, in-order trail of the events emitted for one task. Bounded
/// because a [`TaskEntry`] already lives for the adapter's lifetime; an
/// unbounded log would compound that. The oldest pair is dropped once the ring
/// is full.
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

/// One completed exchange: what was asked, and what the agent answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Turn {
    /// The message posted to the gateway, as the user wrote it — never the
    /// replayed form, or each turn would nest the last.
    pub(crate) user: String,
    /// The agent's reply.
    pub(crate) agent: String,
}

/// A bounded, in-order record of a task's completed exchanges. `POST /webhook`
/// is stateless, so this is the only conversation that exists: a follow-up
/// turn is understood only because the adapter replays this into it. Bounded
/// so a long-lived task cannot grow one request without limit; the oldest turn
/// is dropped once full.
#[derive(Default)]
pub(crate) struct Transcript {
    turns: VecDeque<Turn>,
}

impl Transcript {
    /// Append one completed exchange, evicting the oldest if at capacity.
    pub(crate) fn push(&mut self, turn: Turn) {
        if self.turns.len() == TRANSCRIPT_CAPACITY {
            self.turns.pop_front();
        }
        self.turns.push_back(turn);
    }

    /// A cloned, oldest-first snapshot of the exchanges.
    pub(crate) fn snapshot(&self) -> Vec<Turn> {
        self.turns.iter().cloned().collect()
    }
}

/// One dispatched task as the adapter tracks it: whether a webhook request is
/// currently in flight on its behalf, the bounded trail of what has been
/// emitted for it, and the conversation its next turn replays.
///
/// The owning [`TaskId`] is the map key, so it is not duplicated here. Created
/// by a `Create`, re-armed by an `Update`; a terminal outcome clears
/// [`in_flight`](Self::in_flight) but keeps the entry, so a later update still
/// resolves.
pub(crate) struct TaskEntry {
    /// Whether a `POST /webhook` is executing for this task right now. An
    /// update is only accepted while this is `false` — the webhook API cannot
    /// deliver input to an executing turn.
    pub(crate) in_flight: bool,
    /// The bounded trail of events emitted for this task.
    pub(crate) events: EventRing,
    /// The exchanges completed under this task, replayed into its next turn.
    pub(crate) transcript: Transcript,
}

impl TaskEntry {
    /// A new entry whose first request is departing.
    pub(crate) fn new() -> Self {
        Self {
            in_flight: true,
            events: EventRing::default(),
            transcript: Transcript::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completion(msg: &str) -> DispatchEvent {
        DispatchEvent::Completion {
            task_id: Arc::from("t"),
            message: Arc::from(msg),
        }
    }

    #[test]
    fn minted_ids_are_unique_prefixed_and_session_id_safe() {
        let a = TaskId::mint();
        let b = TaskId::mint();
        assert_ne!(a, b);
        assert!(a.as_str().starts_with("pc-"));
        // The gateway silently drops an id over 128 chars or outside
        // [A-Za-z0-9._-]; a minted id must never trip that.
        assert!(a.as_str().len() <= 128);
        assert!(
            a.as_str()
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
        );
    }

    #[test]
    fn transcript_is_bounded_and_keeps_the_newest() {
        let mut transcript = Transcript::default();
        for i in 0..(TRANSCRIPT_CAPACITY + 3) {
            transcript.push(Turn {
                user: i.to_string(),
                agent: format!("reply {i}"),
            });
        }
        let snap = transcript.snapshot();
        assert_eq!(snap.len(), TRANSCRIPT_CAPACITY);
        // Oldest three (0..=2) were evicted; the replay now starts at 3.
        assert_eq!(snap.first().unwrap().user, "3");
        assert_eq!(
            snap.last().unwrap().user,
            (TRANSCRIPT_CAPACITY + 2).to_string()
        );
    }

    #[test]
    fn ring_is_bounded_and_keeps_the_newest() {
        let mut ring = EventRing::default();
        for i in 0..(EVENT_RING_CAPACITY + 5) {
            ring.push(SystemTime::UNIX_EPOCH, completion(&i.to_string()));
        }
        let snap = ring.snapshot();
        assert_eq!(snap.len(), EVENT_RING_CAPACITY);
        // Oldest five (0..=4) were evicted; the trail now starts at 5.
        assert_eq!(snap.first().unwrap().1, completion("5"));
        assert_eq!(
            snap.last().unwrap().1,
            completion(&(EVENT_RING_CAPACITY + 4).to_string())
        );
    }
}
