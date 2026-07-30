//! [`TelemetrySink`]: a pluggable encoder of finished
//! [`TurnRecord`](crate::TurnRecord)s.
//!
//! One [`TurnAssembler`](crate::TurnAssembler) feeds every sink the same
//! record, so sinks are encodings, not sources of truth: JSONL for a
//! fine-tune dataset, OTLP for dashboards, an in-memory vec for tests. Sinks
//! run on the telemetry consumer (natively the [`Session`](crate::Session)
//! worker thread), never on the pipeline thread, so a sink may block on I/O.

use std::fmt;
use std::sync::Arc;

use crate::record::TurnRecord;

/// A sink failed to record or flush.
#[derive(Clone, Debug)]
pub struct SinkError {
    /// What went wrong.
    pub message: Arc<str>,
}

impl SinkError {
    /// An error with `message`.
    pub fn new(message: impl Into<Arc<str>>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "telemetry sink error: {}", self.message)
    }
}

impl std::error::Error for SinkError {}

/// Encodes finished turn records to wherever they live.
pub trait TelemetrySink {
    /// Record one finished turn.
    fn record(&mut self, turn: &TurnRecord) -> Result<(), SinkError>;

    /// Flush buffered output; called at session end.
    fn flush(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}
