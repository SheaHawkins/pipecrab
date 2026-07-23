//! [`DispatchError`]: the transport-facing error a [`DispatchSource`] or
//! [`DispatchSink`] returns, carrying the recoverable/fatal classification the
//! ingress and egress stages translate into a [`StageError`].
//!
//! [`DispatchSource`]: crate::DispatchSource
//! [`DispatchSink`]: crate::DispatchSink

use std::fmt;
use std::sync::Arc;

use pipecrab_runtime::StageError;

/// Why a transport call ([`DispatchSource::next_event`] or
/// [`DispatchSink::send_command`]) failed.
///
/// Mirrors [`StageError`]: a message plus a `fatal` flag deciding whether the
/// pipeline tears down or carries on. A transport classifies its own errors —
/// a dropped frame is recoverable; a closed socket that cannot reconnect is
/// fatal — and the stage forwards that classification unchanged.
///
/// [`DispatchSource::next_event`]: crate::DispatchSource::next_event
/// [`DispatchSink::send_command`]: crate::DispatchSink::send_command
#[derive(Debug, Clone)]
pub struct DispatchError {
    /// Human-readable description of what went wrong.
    pub message: Arc<str>,
    /// Whether the failure is unrecoverable and the pipeline should shut down.
    pub fatal: bool,
}

impl DispatchError {
    /// A recoverable transport error: the pipeline keeps running.
    pub fn recoverable(message: impl Into<Arc<str>>) -> Self {
        Self {
            message: message.into(),
            fatal: false,
        }
    }

    /// A fatal transport error: the pipeline should shut down.
    pub fn fatal(message: impl Into<Arc<str>>) -> Self {
        Self {
            message: message.into(),
            fatal: true,
        }
    }
}

impl fmt::Display for DispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.fatal {
            "fatal dispatch error"
        } else {
            "dispatch error"
        };
        write!(f, "{kind}: {}", self.message)
    }
}

impl std::error::Error for DispatchError {}

impl From<DispatchError> for StageError {
    /// Carry the transport's own recoverable/fatal classification into the
    /// stage-error the run loop surfaces.
    fn from(error: DispatchError) -> Self {
        if error.fatal {
            StageError::fatal(error.message)
        } else {
            StageError::new(error.message)
        }
    }
}
