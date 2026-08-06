//! JSONL sink for pipecrab telemetry: one self-contained JSON
//! [`TurnRecord`] per line.
//!
//! This is the fine-tune dataset path: every line carries the session id,
//! turn number, user text, agent text, tool calls with correlated results,
//! and the turn's timing/span data — assembled in-process, unsampled,
//! independent of any observability backend's retention or attribute limits.
//!
//! ```no_run
//! use pipecrab_telemetry::Telemetry;
//! use pipecrab_telemetry_jsonl::JsonlSink;
//!
//! let telemetry = Telemetry::builder()
//!     .sink(JsonlSink::create("turns.jsonl").unwrap())
//!     .build();
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use pipecrab_telemetry::{SinkError, TelemetrySink, TurnRecord};

/// Writes one JSON-encoded [`TurnRecord`] per line to any [`Write`].
///
/// Runs on the telemetry worker, never the pipeline thread, so buffered file
/// I/O is fine; each record is flushed on write so a crash loses at most the
/// in-progress line.
pub struct JsonlSink<W: Write> {
    writer: W,
}

impl JsonlSink<BufWriter<File>> {
    /// Create (or truncate) a file at `path` and write records to it.
    pub fn create(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self::new(BufWriter::new(File::create(path)?)))
    }

    /// Append records to the file at `path`, creating it if absent — the
    /// usual choice for a dataset that accumulates across sessions.
    pub fn append(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self::new(BufWriter::new(file)))
    }
}

impl<W: Write> JsonlSink<W> {
    /// A sink over an arbitrary writer.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Consume the sink, returning the writer (tests, in-memory buffers).
    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W: Write> TelemetrySink for JsonlSink<W> {
    fn record(&mut self, turn: &TurnRecord) -> Result<(), SinkError> {
        let mut line =
            serde_json::to_vec(turn).map_err(|e| SinkError::new(format!("encode: {e}")))?;
        line.push(b'\n');
        self.writer
            .write_all(&line)
            .and_then(|()| self.writer.flush())
            .map_err(|e| SinkError::new(format!("write: {e}")))
    }

    fn flush(&mut self) -> Result<(), SinkError> {
        self.writer
            .flush()
            .map_err(|e| SinkError::new(format!("flush: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use pipecrab_telemetry::{SessionInfo, ToolCallRecord, TurnEnd, TurnOrigin, TurnTimings};

    fn record() -> TurnRecord {
        TurnRecord {
            session: SessionInfo {
                id: Arc::from("s1"),
                started_unix_ms: 1_700_000_000_000,
            },
            turn: 3,
            origin: TurnOrigin::Speech,
            end: TurnEnd::NextTurn,
            started_ms: 100.0,
            ended_ms: 2_000.0,
            user_text: Some(Arc::from("what's the weather in paris")),
            agent_text: Some(Arc::from("It's sunny and 18 degrees.")),
            tool_calls: vec![ToolCallRecord {
                id: Arc::from("call-1"),
                name: Arc::from("weather"),
                arguments_json: Arc::from(r#"{"city":"paris"}"#),
                result: Some(Arc::from("sunny, 18C")),
                at_ms: 950.0,
            }],
            interrupted_at_ms: None,
            timings: TurnTimings {
                user_audio_secs: Some(1.5),
                stt_ms: Some(100.0),
                lm_ttft_ms: Some(180.0),
                lm_total_ms: Some(730.0),
                time_to_first_speech_ms: Some(910.0),
                response_latency_ms: Some(900.0),
                speech_before_interrupt_ms: None,
            },
            stages: Vec::new(),
            errors: Vec::new(),
            lost_events: 0,
        }
    }

    #[test]
    fn writes_one_parseable_json_line_per_record() {
        let mut sink = JsonlSink::new(Vec::new());
        sink.record(&record()).unwrap();
        sink.record(&record()).unwrap();
        let out = sink.into_inner();
        let lines: Vec<&[u8]> = out
            .split(|b| *b == b'\n')
            .filter(|l| !l.is_empty())
            .collect();
        assert_eq!(lines.len(), 2);
        let parsed: serde_json::Value = serde_json::from_slice(lines[0]).unwrap();
        assert_eq!(parsed["session"]["id"], "s1");
        assert_eq!(parsed["turn"], 3);
        assert_eq!(parsed["origin"], "speech");
        assert_eq!(parsed["user_text"], "what's the weather in paris");
        assert_eq!(parsed["tool_calls"][0]["result"], "sunny, 18C");
        assert_eq!(parsed["timings"]["time_to_first_speech_ms"], 910.0);
    }
}
