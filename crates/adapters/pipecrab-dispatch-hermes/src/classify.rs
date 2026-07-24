//! The pure Hermes-status → dispatch-outcome mapping.
//!
//! [`classify`] is a total function of a run's observed `(status, output,
//! error)` alone — no registry, no I/O — so the interesting policy (which
//! statuses are terminal, what each projects to) is unit-testable in isolation.
//! Turning a [`Classified`] into an actual [`DispatchEvent`] and applying its
//! registry effect (clear the active run, dedupe progress) is the poller's job.

/// The message used when a completed run carries no output.
const NO_OUTPUT: &str = "(no output)";
/// The message for a `failed` run whose body offers no error detail.
const FAILED_FALLBACK: &str = "run failed";

/// What one poll observation of a run means, independent of registry state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Classified {
    /// Terminal success — projects to `Completion` and clears the active run.
    Completed {
        /// The run's output, or a `(no output)` placeholder.
        message: String,
    },
    /// Terminal failure — projects to a non-retryable `Failure`.
    Failed {
        /// Best-effort error detail from the run body.
        message: String,
    },
    /// Terminally cancelled out-of-band (e.g. via the dashboard or `/stop`) —
    /// projects to a *retryable* `Failure`, since re-dispatching is coherent.
    Cancelled,
    /// Not terminal — projects to a `Progress` update, but only when the status
    /// string has actually changed.
    InFlight {
        /// The current status string, forwarded as the progress detail.
        status: String,
    },
}

/// Map a run's observed state to its [`Classified`] outcome.
///
/// Terminal statuses are `completed` / `failed` / `cancelled`. Everything else —
/// `started`, `running`, `stopping`, and any status string the docs don't
/// enumerate — is treated as in-flight and forwarded as progress, so a Hermes
/// build that adds a new transitional status stays forward-compatible.
pub(crate) fn classify(status: &str, output: Option<&str>, error: Option<&str>) -> Classified {
    match status {
        "completed" => Classified::Completed {
            message: non_empty(output).unwrap_or(NO_OUTPUT).to_owned(),
        },
        "failed" => Classified::Failed {
            // Hermes gives no documented error field; fall back through the
            // candidate detail (extracted upstream), then output, then a
            // generic message.
            message: non_empty(error)
                .or_else(|| non_empty(output))
                .unwrap_or(FAILED_FALLBACK)
                .to_owned(),
        },
        "cancelled" => Classified::Cancelled,
        other => Classified::InFlight {
            status: other.to_owned(),
        },
    }
}

/// `Some(s)` only when `s` is present and not empty.
fn non_empty(s: Option<&str>) -> Option<&str> {
    s.filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_uses_output_or_placeholder() {
        assert_eq!(
            classify("completed", Some("2+2 = 4"), None),
            Classified::Completed {
                message: "2+2 = 4".to_owned()
            }
        );
        assert_eq!(
            classify("completed", None, None),
            Classified::Completed {
                message: NO_OUTPUT.to_owned()
            }
        );
        assert_eq!(
            classify("completed", Some(""), None),
            Classified::Completed {
                message: NO_OUTPUT.to_owned()
            }
        );
    }

    #[test]
    fn failed_prefers_error_then_output_then_fallback() {
        assert_eq!(
            classify("failed", None, Some("model exploded")),
            Classified::Failed {
                message: "model exploded".to_owned()
            }
        );
        assert_eq!(
            classify("failed", Some("partial"), None),
            Classified::Failed {
                message: "partial".to_owned()
            }
        );
        assert_eq!(
            classify("failed", None, None),
            Classified::Failed {
                message: FAILED_FALLBACK.to_owned()
            }
        );
    }

    #[test]
    fn cancelled_is_its_own_outcome() {
        assert_eq!(classify("cancelled", None, None), Classified::Cancelled);
    }

    #[test]
    fn non_terminal_and_unknown_statuses_are_in_flight() {
        for s in [
            "started",
            "running",
            "stopping",
            "queued",
            "some_new_status",
        ] {
            assert_eq!(
                classify(s, None, None),
                Classified::InFlight {
                    status: s.to_owned()
                }
            );
        }
    }
}
