//! [`TurnRecord`]: the canonical, self-contained record of one turn — the
//! single source of truth every [`TelemetrySink`](crate::TelemetrySink)
//! encodes. Times are `f64` milliseconds; per-turn instants are offsets in
//! the session clock's timebase, anchored by
//! [`SessionInfo::started_unix_ms`].

use std::sync::Arc;

/// Identity of one session (an "interaction": one pipeline run).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SessionInfo {
    /// Application-chosen or generated session id; on every record.
    pub id: Arc<str>,
    /// Wall-clock anchor: Unix time in ms at the session clock's epoch.
    pub started_unix_ms: u64,
}

/// Why a turn ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum TurnEnd {
    /// The next turn opened.
    NextTurn,
    /// A barge-in `Interrupt` cut it short.
    Interrupted,
    /// The session ended (`Stop` or shutdown) while the turn was open.
    SessionEnd,
}

/// What opened the turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum TurnOrigin {
    /// The user spoke (`SpeechStarted`).
    Speech,
    /// The model spoke unprompted (`GenerationStarted` with no open turn —
    /// e.g. a dispatch event coming home).
    Model,
}

/// One tool call and, when it came home within the session, its result.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ToolCallRecord {
    /// The call id correlating call and result.
    pub id: Arc<str>,
    /// Model-facing tool name.
    pub name: Arc<str>,
    /// Complete argument object as JSON text, verbatim from the model adapter.
    pub arguments_json: Arc<str>,
    /// The tool's result text, once its `ToolResult` re-entered the pipeline.
    /// `None` for calls still in flight when the record closed.
    pub result: Option<Arc<str>>,
    /// Offset (ms) at which the call was observed.
    pub at_ms: f64,
}

/// The turn-level latency metrics that are only clean to measure in-process.
/// All `None` when the turn never reached that point.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TurnTimings {
    /// Seconds of user utterance audio (sample-counted, not wall time).
    pub user_audio_secs: Option<f64>,
    /// `SpeechStopped` to the user-final transcript: STT latency as felt.
    pub stt_ms: Option<f64>,
    /// `GenerationStarted` to the first agent partial: LM time-to-first-token.
    pub lm_ttft_ms: Option<f64>,
    /// `GenerationStarted` to `GenerationFinished`.
    pub lm_total_ms: Option<f64>,
    /// Last user utterance audio chunk to the first agent audio at the tail:
    /// time-to-first-speech as the user hears it, VAD hangover excluded.
    pub time_to_first_speech_ms: Option<f64>,
    /// `SpeechStopped` to the first agent audio at the tail. Exceeds
    /// `time_to_first_speech_ms` by roughly the VAD hangover
    /// (`min_silence_duration`); the difference is VAD-tuning headroom.
    pub response_latency_ms: Option<f64>,
}

/// Per-stage decide/perform aggregates for one turn: the turn's spans.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct StageTimings {
    /// The stage's wiring path (e.g. `"3"`, `"2.0"`).
    pub path: Arc<str>,
    /// The stage's name.
    pub name: Arc<str>,
    /// Data frames decided.
    pub frames: u64,
    /// Total time in `decide_*` (orchestrator occupancy), ms.
    pub decide_ms: f64,
    /// Largest single decide, ms.
    pub decide_max_ms: f64,
    /// Effects performed (completed, failed, or aborted).
    pub performs: u64,
    /// Total time in `perform`, ms.
    pub perform_ms: f64,
    /// Largest single perform, ms.
    pub perform_max_ms: f64,
    /// Performs that returned an error.
    pub errors: u64,
    /// Performs dropped by a barge-in.
    pub aborted: u64,
}

/// One assembled turn: a trace, with everything a dashboard span tree or a
/// fine-tune training row needs.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TurnRecord {
    /// The owning session.
    pub session: SessionInfo,
    /// 0-based turn number within the session.
    pub turn: u64,
    /// What opened the turn.
    pub origin: TurnOrigin,
    /// Why it closed.
    pub end: TurnEnd,
    /// Open offset (ms since session start).
    pub started_ms: f64,
    /// Close offset (ms since session start).
    pub ended_ms: f64,
    /// The user's final utterance transcript, if one was produced.
    pub user_text: Option<Arc<str>>,
    /// The agent's final reply text, if a generation produced one.
    pub agent_text: Option<Arc<str>>,
    /// Tool calls the turn's generation emitted, results correlated by id.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Latency metrics measured in-process.
    pub timings: TurnTimings,
    /// Per-stage spans, in wiring order.
    pub stages: Vec<StageTimings>,
    /// Distinct pipeline error messages observed during the turn.
    pub errors: Vec<Arc<str>>,
    /// Telemetry events lost to channel overflow during the turn; when
    /// nonzero the record may be incomplete.
    pub lost_events: u64,
}
