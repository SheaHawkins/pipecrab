//! The pure webhook-outcome → dispatch-outcome mapping.
//!
//! [`classify`] is a total function of one request's [`PostResult`] alone — no
//! registry, no I/O — so the interesting policy (which statuses fail, which
//! failures are worth retrying) is unit-testable in isolation. Turning a
//! [`Classified`] into an actual [`DispatchEvent`](pipecrab_core::DispatchEvent)
//! and applying its registry effect is the worker's job.

use crate::client::{PostResult, ReplyBody};

/// The message used when a 2xx reply carries no `response`.
const NO_REPLY: &str = "(no reply)";

/// What one webhook outcome means, independent of registry state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Classified {
    /// The turn ran and answered — projects to `Completion`.
    Completed {
        /// The agent's reply, or a `(no reply)` placeholder.
        message: String,
    },
    /// The turn did not answer — projects to a `Failure`.
    Failed {
        /// What went wrong, from the error envelope or the transport.
        message: String,
        /// Whether re-dispatching might succeed.
        retryable: bool,
    },
}

/// Map one request's outcome to its [`Classified`] result.
///
/// The policy: any 2xx answered (a dedupe-`duplicate` body is the one 2xx that
/// didn't — the original reply is unrecoverable, so it fails retryably); a 4xx
/// is the caller's fault and retrying verbatim cannot help (429 excepted — the
/// limit resets); a 5xx or a transport error is the gateway's and might (the
/// `needs_quickstart` 503 excepted — an unconfigured gateway stays that way
/// until an operator acts).
pub(crate) fn classify(outcome: &PostResult) -> Classified {
    match outcome {
        PostResult::Response { status, body } => classify_response(*status, body),
        PostResult::Transport { message } => Classified::Failed {
            // The request died in transit *or* while the turn was executing —
            // the gateway may still have finished it, but its reply is lost.
            message: format!("{message} (the turn may still have run; its reply is lost)"),
            retryable: true,
        },
    }
}

fn classify_response(status: u16, body: &ReplyBody) -> Classified {
    if (200..300).contains(&status) {
        if body.status.as_deref() == Some("duplicate") {
            return Classified::Failed {
                message: "the gateway had already processed this idempotency key; \
                          the original reply is unrecoverable"
                    .to_owned(),
                retryable: true,
            };
        }
        return Classified::Completed {
            message: non_empty(body.response.as_deref())
                .unwrap_or(NO_REPLY)
                .to_owned(),
        };
    }

    let detail = non_empty(body.error.as_deref()).unwrap_or("(no detail)");
    let retryable = match status {
        // The limit window resets on its own.
        429 => true,
        400..=499 => false,
        // A gateway booted without a model stays that way until an operator
        // completes /quickstart; anything else server-side may recover.
        _ => !detail.contains("needs_quickstart"),
    };
    Classified::Failed {
        message: format!("zeroclaw webhook returned {status}: {detail}"),
        retryable,
    }
}

/// `Some(s)` only when `s` is present and not empty.
fn non_empty(s: Option<&str>) -> Option<&str> {
    s.filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(status: u16, body: ReplyBody) -> PostResult {
        PostResult::Response { status, body }
    }

    fn reply(text: &str) -> ReplyBody {
        ReplyBody {
            response: Some(text.to_owned()),
            error: None,
            status: None,
        }
    }

    fn error(text: &str) -> ReplyBody {
        ReplyBody {
            response: None,
            error: Some(text.to_owned()),
            status: None,
        }
    }

    #[test]
    fn a_reply_completes_and_an_empty_one_gets_a_placeholder() {
        assert_eq!(
            classify(&response(200, reply("2+2 = 4"))),
            Classified::Completed {
                message: "2+2 = 4".to_owned()
            }
        );
        assert_eq!(
            classify(&response(200, reply(""))),
            Classified::Completed {
                message: NO_REPLY.to_owned()
            }
        );
        assert_eq!(
            classify(&response(200, ReplyBody::default())),
            Classified::Completed {
                message: NO_REPLY.to_owned()
            }
        );
    }

    #[test]
    fn a_duplicate_dedupe_is_a_retryable_failure() {
        let dup = ReplyBody {
            response: None,
            error: None,
            status: Some("duplicate".to_owned()),
        };
        match classify(&response(200, dup)) {
            Classified::Failed {
                message, retryable, ..
            } => {
                assert!(retryable);
                assert!(message.contains("idempotency"), "message: {message}");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn client_errors_are_final_but_rate_limits_are_not() {
        for status in [400, 401, 404] {
            match classify(&response(status, error("nope"))) {
                Classified::Failed {
                    message, retryable, ..
                } => {
                    assert!(!retryable, "{status} must not be retryable");
                    assert!(message.contains(&status.to_string()));
                    assert!(message.contains("nope"));
                }
                other => panic!("expected Failed for {status}, got {other:?}"),
            }
        }
        assert_eq!(
            classify(&response(429, error("Too many webhook requests"))),
            Classified::Failed {
                message: "zeroclaw webhook returned 429: Too many webhook requests".to_owned(),
                retryable: true,
            }
        );
    }

    #[test]
    fn server_errors_retry_except_an_unconfigured_gateway() {
        match classify(&response(500, error("LLM request failed"))) {
            Classified::Failed { retryable, .. } => assert!(retryable),
            other => panic!("expected Failed, got {other:?}"),
        }
        match classify(&response(503, error("needs_quickstart"))) {
            Classified::Failed { retryable, .. } => {
                assert!(!retryable, "an unconfigured gateway will not fix itself");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
        // A 503 with no detail is an overloaded gateway, not an unconfigured one.
        match classify(&response(503, ReplyBody::default())) {
            Classified::Failed { retryable, .. } => assert!(retryable),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn a_transport_error_is_retryable_and_flags_the_lost_reply() {
        let outcome = PostResult::Transport {
            message: "connection reset".to_owned(),
        };
        match classify(&outcome) {
            Classified::Failed {
                message, retryable, ..
            } => {
                assert!(retryable);
                assert!(message.contains("connection reset"));
                assert!(message.contains("reply is lost"), "message: {message}");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }
}
