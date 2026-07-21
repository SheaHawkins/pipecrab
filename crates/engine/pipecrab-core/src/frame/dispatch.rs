//! Native asynchronous-dispatch protocol frames: commands that drive
//! long-running external tasks and events reporting their state. Every dispatch
//! frame survives an interrupt flush — external task state outlives the turn.

use std::sync::Arc;

/// A command driving an asynchronous external task. `tool_call_id` names the
/// model invocation; `task_id` names the task and exists only once a
/// [`Create`](DispatchCommand::Create) is accepted (see [`DispatchEvent::Accepted`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchCommand {
    /// Start a new task; no task ID exists yet.
    Create {
        /// The [`ToolCall::id`](crate::ToolCall::id) that requested the task.
        tool_call_id: Arc<str>,
        /// The work to perform.
        task: Arc<str>,
        /// Optional extra context.
        context: Option<Arc<str>>,
    },
    /// Send a follow-up to an already-accepted task.
    Update {
        /// The [`ToolCall::id`](crate::ToolCall::id) that requested the update.
        tool_call_id: Arc<str>,
        /// The accepted task's identifier (from [`DispatchEvent::Accepted`]).
        task_id: Arc<str>,
        /// The follow-up message.
        message: Arc<str>,
    },
}

/// An event reporting an asynchronous task's state.
/// [`Rejected`](DispatchEvent::Rejected) is a pre-task failure (carries
/// `tool_call_id`); [`Failure`](DispatchEvent::Failure) is a
/// post-[`Accepted`](DispatchEvent::Accepted) failure (carries `task_id`).
/// [`Progress`](DispatchEvent::Progress) is informational.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchEvent {
    /// A [`DispatchCommand::Create`] was accepted; the task now has an ID.
    Accepted {
        /// The [`ToolCall::id`](crate::ToolCall::id) from the create command.
        tool_call_id: Arc<str>,
        /// The new task's identifier.
        task_id: Arc<str>,
    },
    /// A command failed before a task ID was assigned.
    Rejected {
        /// The [`ToolCall::id`](crate::ToolCall::id) from the rejected command.
        tool_call_id: Arc<str>,
        /// Human-readable reason for the rejection.
        message: Arc<str>,
    },
    /// An informational progress update from an accepted task.
    Progress {
        /// The task this update concerns.
        task_id: Arc<str>,
        /// The progress detail.
        message: Arc<str>,
    },
    /// An accepted task is asking a question.
    Question {
        /// The task asking the question.
        task_id: Arc<str>,
        /// The question text.
        message: Arc<str>,
    },
    /// An accepted task finished successfully.
    Completion {
        /// The task that completed.
        task_id: Arc<str>,
        /// The completion detail.
        message: Arc<str>,
    },
    /// A previously accepted task later failed.
    Failure {
        /// The task that failed.
        task_id: Arc<str>,
        /// Human-readable reason for the failure.
        message: Arc<str>,
        /// Whether retrying the task might succeed.
        retryable: bool,
    },
}

/// A dispatch protocol frame: a [`Command`](DispatchFrame::Command) requests
/// work, an [`Event`](DispatchFrame::Event) reports what happened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchFrame {
    /// A task-state update; see [`DispatchEvent`].
    Event(DispatchEvent),
    /// A request that drives a task; see [`DispatchCommand`].
    Command(DispatchCommand),
}

impl From<DispatchEvent> for DispatchFrame {
    /// Wrap a [`DispatchEvent`] as [`DispatchFrame::Event`].
    fn from(event: DispatchEvent) -> Self {
        DispatchFrame::Event(event)
    }
}

impl From<DispatchCommand> for DispatchFrame {
    /// Wrap a [`DispatchCommand`] as [`DispatchFrame::Command`].
    fn from(command: DispatchCommand) -> Self {
        DispatchFrame::Command(command)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataFrame;

    #[test]
    fn subtypes_construct_and_compare() {
        let create = DispatchCommand::Create {
            tool_call_id: Arc::from("call-1"),
            task: Arc::from("book flight"),
            context: None,
        };
        assert_eq!(create.clone(), create);
        // A differing `context` makes two otherwise-equal creates distinct.
        assert_ne!(
            create,
            DispatchCommand::Create {
                tool_call_id: Arc::from("call-1"),
                task: Arc::from("book flight"),
                context: Some(Arc::from("window seat")),
            }
        );

        // `retryable` participates in equality.
        let fail = DispatchEvent::Failure {
            task_id: Arc::from("task-7"),
            message: Arc::from("timeout"),
            retryable: true,
        };
        assert_ne!(
            fail,
            DispatchEvent::Failure {
                task_id: Arc::from("task-7"),
                message: Arc::from("timeout"),
                retryable: false,
            }
        );
    }

    #[test]
    fn tool_call_id_and_task_id_are_distinct_fields() {
        // Accepted is the hinge from a pre-task `tool_call_id` to the assigned
        // `task_id`; they are independent values that need not match.
        let accepted = DispatchEvent::Accepted {
            tool_call_id: Arc::from("call-1"),
            task_id: Arc::from("task-42"),
        };
        let DispatchEvent::Accepted {
            tool_call_id,
            task_id,
        } = &accepted
        else {
            unreachable!()
        };
        assert_ne!(tool_call_id, task_id);

        // Rejected fails before a task id exists, so it carries only the call id.
        let rejected = DispatchEvent::Rejected {
            tool_call_id: Arc::from("call-1"),
            message: Arc::from("quota exceeded"),
        };
        assert_ne!(accepted, rejected);
    }

    #[test]
    fn events_and_commands_convert_into_dispatch_frame() {
        let event = DispatchEvent::Progress {
            task_id: Arc::from("task-1"),
            message: Arc::from("50%"),
        };
        assert_eq!(
            DispatchFrame::from(event.clone()),
            DispatchFrame::Event(event)
        );

        let command = DispatchCommand::Update {
            tool_call_id: Arc::from("call-2"),
            task_id: Arc::from("task-1"),
            message: Arc::from("hurry"),
        };
        assert_eq!(
            DispatchFrame::from(command.clone()),
            DispatchFrame::Command(command)
        );
    }

    #[test]
    fn dispatch_frame_rides_the_data_lane() {
        let frame = DispatchFrame::Event(DispatchEvent::Completion {
            task_id: Arc::from("task-1"),
            message: Arc::from("done"),
        });
        match DataFrame::from(frame.clone()) {
            DataFrame::Dispatch(f) => assert_eq!(f, frame),
            other => panic!("expected Dispatch, got {other:?}"),
        }
    }

    #[test]
    fn every_dispatch_frame_survives_flush() {
        // External task state must outlive a barge-in — commands and every event
        // variant survive the data-lane flush.
        let events = [
            DispatchEvent::Accepted {
                tool_call_id: Arc::from("c"),
                task_id: Arc::from("t"),
            },
            DispatchEvent::Rejected {
                tool_call_id: Arc::from("c"),
                message: Arc::from("no"),
            },
            DispatchEvent::Progress {
                task_id: Arc::from("t"),
                message: Arc::from("p"),
            },
            DispatchEvent::Question {
                task_id: Arc::from("t"),
                message: Arc::from("q"),
            },
            DispatchEvent::Completion {
                task_id: Arc::from("t"),
                message: Arc::from("c"),
            },
            DispatchEvent::Failure {
                task_id: Arc::from("t"),
                message: Arc::from("f"),
                retryable: false,
            },
        ];
        for event in events {
            assert!(DataFrame::from(DispatchFrame::from(event)).survives_flush());
        }

        let commands = [
            DispatchCommand::Create {
                tool_call_id: Arc::from("c"),
                task: Arc::from("do"),
                context: None,
            },
            DispatchCommand::Update {
                tool_call_id: Arc::from("c"),
                task_id: Arc::from("t"),
                message: Arc::from("m"),
            },
        ];
        for command in commands {
            assert!(DataFrame::from(DispatchFrame::from(command)).survives_flush());
        }
    }
}
