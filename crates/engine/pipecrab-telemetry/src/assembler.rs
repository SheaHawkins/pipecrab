//! [`TurnAssembler`]: folds the raw [`Event`] stream into
//! [`TurnRecord`]s.
//!
//! # How assembly works
//!
//! A frame that *forwards* is observed once per boundary it crosses, so the
//! same `SpeechStarted` reaches the assembler several times. Rather than
//! thread ids through frames, the assembler exploits the pipeline's fixed
//! topology: **the first boundary a frame kind is observed at becomes that
//! kind's designated boundary** for the rest of the session, and observations
//! of the kind elsewhere are propagation duplicates, ignored. (Tool calls
//! carry ids and are deduped by id instead.) The first boundary is where the
//! emitting stage's output lands, so it is also the earliest — and least
//! queue-skewed — timestamp available.
//!
//! # Turn lifecycle
//!
//! A turn opens on `SpeechStarted` ([`TurnOrigin::Speech`]) or, when the model
//! speaks unprompted (a dispatch result coming home), on `GenerationStarted`
//! with no turn open ([`TurnOrigin::Model`]). It closes when the next turn
//! opens, an `Interrupt` cuts it short, or `Stop`/session end flushes it. A
//! turn may span several generations (tool-call round trips):
//! `lm_total_ms` measures first `GenerationStarted` to last
//! `GenerationFinished`.
//!
//! An `Interrupt` closes a turn as [`TurnEnd::Interrupted`] only when it
//! actually cut the agent off — a generation still open, or agent audio at
//! the tail within the last second. A barge-in originator fires on every
//! speech onset, agent idle or not, so an idle-time `Interrupt` is treated as
//! the next utterance's herald and leaves the turn open. Interrupted records
//! carry `interrupted_at_ms` and `speech_before_interrupt_ms` (how long the
//! agent got to speak).
//!
//! # Attribution limits
//!
//! Per-stage spans aggregate only while a turn is open, so upstream work that
//! precedes `SpeechStarted` (capture resampling, VAD scoring) lands outside
//! them. A tool result arriving after its turn closed is not retroactively
//! attached — the already-emitted record keeps `result: None`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use pipecrab_core::{Finality, Role, SystemFrame};
use pipecrab_runtime::{PerformOutcome, StageId, TAIL_STAGE};

use crate::event::{Event, EventKind, FrameInfo};
use crate::record::{
    SessionInfo, StageTimings, ToolCallRecord, TurnEnd, TurnOrigin, TurnRecord, TurnTimings,
};

/// How recently agent audio must have crossed the tail for an `Interrupt` to
/// count as cutting the agent off (covers the sink ring depth and pump pacing
/// between the tail tap and the speaker).
const SPEAKING_GRACE: Duration = Duration::from_millis(1_000);

/// How long a turn whose generation has finished must stay quiet (no
/// turn-relevant frames) before [`TurnAssembler::tick`] closes it as
/// [`TurnEnd::Completed`]. Long enough to bridge synthesis gaps between the
/// reply's final sentences; short enough that a record lands promptly after
/// the agent stops speaking.
const QUIET_CLOSE: Duration = Duration::from_millis(1_500);

/// The frame kinds that learn a designated boundary; see the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Learned {
    SpeechEdge,
    UserFinal,
    AgentPartial,
    AgentFinal,
    Generation,
}

#[derive(Default)]
struct StageAgg {
    name: Option<Arc<str>>,
    frames: u64,
    decide: Duration,
    decide_max: Duration,
    performs: u64,
    perform: Duration,
    perform_max: Duration,
    errors: u64,
    aborted: u64,
}

struct TurnState {
    origin: TurnOrigin,
    started_at: Duration,
    /// When a turn-relevant frame was last observed; drives the quiet close.
    /// Deliberately not refreshed by idle capture audio upstream of the VAD.
    last_activity_at: Duration,
    speech_stopped_at: Option<Duration>,
    last_user_audio_at: Option<Duration>,
    user_audio_secs: f64,
    user_text: Option<Arc<str>>,
    user_final_at: Option<Duration>,
    generation_started_at: Option<Duration>,
    first_agent_partial_at: Option<Duration>,
    generation_finished_at: Option<Duration>,
    first_speech_at: Option<Duration>,
    last_speech_at: Option<Duration>,
    agent_finals: Vec<Arc<str>>,
    tool_calls: Vec<ToolCallRecord>,
    errors: Vec<Arc<str>>,
    stages: HashMap<Arc<str>, StageAgg>,
    lost_at_open: u64,
}

impl TurnState {
    fn new(origin: TurnOrigin, at: Duration, lost_at_open: u64) -> Self {
        Self {
            origin,
            started_at: at,
            last_activity_at: at,
            speech_stopped_at: None,
            last_user_audio_at: None,
            user_audio_secs: 0.0,
            user_text: None,
            user_final_at: None,
            generation_started_at: None,
            first_agent_partial_at: None,
            generation_finished_at: None,
            first_speech_at: None,
            last_speech_at: None,
            agent_finals: Vec::new(),
            tool_calls: Vec::new(),
            errors: Vec::new(),
            stages: HashMap::new(),
            lost_at_open,
        }
    }
}

/// Folds [`Event`]s into [`TurnRecord`]s; see the module docs for the rules.
///
/// Feed events in stream order via [`push`](TurnAssembler::push); each call
/// returns at most one turn that just closed. Call
/// [`finish`](TurnAssembler::finish) at session end to flush an open turn.
pub struct TurnAssembler {
    session: SessionInfo,
    turn_seq: u64,
    tail_path: Option<Arc<str>>,
    learned: HashMap<Learned, Arc<str>>,
    current: Option<TurnState>,
    pending_decide: HashMap<Arc<str>, Duration>,
    pending_perform: HashMap<Arc<str>, Duration>,
    lost_total: u64,
    last_at: Duration,
}

impl TurnAssembler {
    /// An assembler stamping records with `session`.
    pub fn new(session: SessionInfo) -> Self {
        Self {
            session,
            turn_seq: 0,
            tail_path: None,
            learned: HashMap::new(),
            current: None,
            pending_decide: HashMap::new(),
            pending_perform: HashMap::new(),
            lost_total: 0,
            last_at: Duration::ZERO,
        }
    }

    /// Fold one event; returns the turn it closed, if any.
    pub fn push(&mut self, event: Event) -> Option<TurnRecord> {
        self.last_at = self.last_at.max(event.at);
        let at = event.at;
        match event.kind {
            EventKind::Wired { stages } => {
                self.learn_tail(&stages);
                None
            }
            EventKind::Lost { count } => {
                self.lost_total += count;
                None
            }
            EventKind::FrameIn(frame) => {
                let stage = event.stage?;
                self.pending_decide.insert(stage.path.clone(), at);
                self.frame(&stage, frame, at)
            }
            EventKind::Decided { .. } => {
                let stage = event.stage?;
                if let Some(t0) = self.pending_decide.remove(&stage.path) {
                    self.aggregate(
                        &stage,
                        |agg, dt| {
                            agg.frames += 1;
                            agg.decide += dt;
                            agg.decide_max = agg.decide_max.max(dt);
                        },
                        at,
                        t0,
                    );
                }
                None
            }
            EventKind::SysIn { frame, .. } => {
                let stage = event.stage?;
                self.system(&stage, frame, at)
            }
            EventKind::SysDecided { .. } => None,
            EventKind::PerformStart => {
                let stage = event.stage?;
                self.pending_perform.insert(stage.path.clone(), at);
                None
            }
            EventKind::PerformEnd { outcome } => {
                let stage = event.stage?;
                if let Some(t0) = self.pending_perform.remove(&stage.path) {
                    self.aggregate(
                        &stage,
                        |agg, dt| {
                            agg.performs += 1;
                            agg.perform += dt;
                            agg.perform_max = agg.perform_max.max(dt);
                            match outcome {
                                PerformOutcome::Ok => {}
                                PerformOutcome::Err { .. } => agg.errors += 1,
                                PerformOutcome::Aborted => agg.aborted += 1,
                            }
                        },
                        at,
                        t0,
                    );
                }
                None
            }
        }
    }

    /// Flush the open turn (if any) as [`TurnEnd::SessionEnd`].
    pub fn finish(&mut self) -> Option<TurnRecord> {
        let at = self.last_at;
        self.close(TurnEnd::SessionEnd, at)
    }

    /// Quiet close: called periodically (with the session clock's `now`) so a
    /// finished turn's record lands without waiting for the next turn to open.
    /// Closes the open turn as [`TurnEnd::Completed`] once its generation has
    /// finished and no turn-relevant frame has been observed for
    /// [`QUIET_CLOSE`]; the record's end time is the last activity, not the
    /// tick. A turn that never generated (a gated noise trigger) is left for
    /// the next turn or session end to close.
    pub fn tick(&mut self, now: Duration) -> Option<TurnRecord> {
        let turn = self.current.as_ref()?;
        let generated =
            turn.generation_started_at.is_some() && turn.generation_finished_at.is_some();
        if !generated || now.saturating_sub(turn.last_activity_at) < QUIET_CLOSE {
            return None;
        }
        let at = turn.last_activity_at;
        self.close(TurnEnd::Completed, at)
    }

    fn learn_tail(&mut self, stages: &[StageId]) {
        if self.tail_path.is_some() {
            return;
        }
        // The first wired report is the top-level pipeline; its tail tap (or,
        // absent one, its last stage) is where frames leave the pipeline.
        let tail = stages
            .iter()
            .find(|s| s.name == TAIL_STAGE)
            .or_else(|| stages.last());
        self.tail_path = tail.map(|s| s.path.clone());
    }

    /// The designated boundary for `kind`: learned from `path` on first sight,
    /// then matched against.
    fn at_learned(&mut self, kind: Learned, path: &Arc<str>) -> bool {
        match self.learned.get(&kind) {
            Some(learned) => **learned == **path,
            None => {
                self.learned.insert(kind, path.clone());
                true
            }
        }
    }

    fn aggregate(
        &mut self,
        stage: &StageId,
        f: impl FnOnce(&mut StageAgg, Duration),
        at: Duration,
        t0: Duration,
    ) {
        let Some(turn) = self.current.as_mut() else {
            return;
        };
        let dt = at.saturating_sub(t0);
        let agg = turn.stages.entry(stage.path.clone()).or_default();
        if agg.name.is_none() {
            agg.name = Some(Arc::from(stage.name));
        }
        f(agg, dt);
    }

    fn frame(&mut self, stage: &StageId, frame: FrameInfo, at: Duration) -> Option<TurnRecord> {
        let path = &stage.path;
        match frame {
            FrameInfo::SpeechStarted => {
                if !self.at_learned(Learned::SpeechEdge, path) {
                    return None;
                }
                let closed = self.close(TurnEnd::NextTurn, at);
                self.open(TurnOrigin::Speech, at);
                closed
            }
            FrameInfo::SpeechStopped => {
                if self.at_learned(Learned::SpeechEdge, path)
                    && let Some(turn) = self.current.as_mut()
                {
                    turn.speech_stopped_at.get_or_insert(at);
                    turn.last_activity_at = at;
                }
                None
            }
            FrameInfo::Audio {
                samples,
                sample_rate,
                channels,
            } => {
                if self
                    .learned
                    .get(&Learned::SpeechEdge)
                    .is_some_and(|p| **p == **path)
                {
                    // Utterance audio: between the edges at the edge boundary.
                    if let Some(turn) = self.current.as_mut()
                        && turn.speech_stopped_at.is_none()
                        && turn.generation_started_at.is_none()
                    {
                        let frames = samples / usize::from(channels.max(1));
                        turn.user_audio_secs += frames as f64 / f64::from(sample_rate.max(1));
                        turn.last_user_audio_at = Some(at);
                        turn.last_activity_at = at;
                    }
                } else if self.tail_path.as_ref().is_some_and(|p| **p == **path)
                    && let Some(turn) = self.current.as_mut()
                    && turn.generation_started_at.is_some()
                {
                    // Agent speech leaving the pipeline.
                    turn.first_speech_at.get_or_insert(at);
                    turn.last_speech_at = Some(at);
                    turn.last_activity_at = at;
                }
                None
            }
            FrameInfo::Transcript {
                text,
                role,
                finality,
                ..
            } => {
                match (role, finality) {
                    (Role::User, Finality::Final) => {
                        if self.at_learned(Learned::UserFinal, path)
                            && let Some(turn) = self.current.as_mut()
                        {
                            turn.user_text = text;
                            turn.user_final_at.get_or_insert(at);
                            turn.last_activity_at = at;
                        }
                    }
                    (Role::Agent, Finality::Partial { .. }) => {
                        if self.at_learned(Learned::AgentPartial, path)
                            && let Some(turn) = self.current.as_mut()
                            && turn.generation_started_at.is_some()
                        {
                            turn.first_agent_partial_at.get_or_insert(at);
                            turn.last_activity_at = at;
                        }
                    }
                    (Role::Agent, Finality::Final) => {
                        if self.at_learned(Learned::AgentFinal, path)
                            && let Some(turn) = self.current.as_mut()
                            && let Some(text) = text
                        {
                            turn.agent_finals.push(text);
                            turn.last_activity_at = at;
                        }
                    }
                    _ => {}
                }
                None
            }
            FrameInfo::GenerationStarted => {
                if !self.at_learned(Learned::Generation, path) {
                    return None;
                }
                if self.current.is_none() {
                    self.open(TurnOrigin::Model, at);
                }
                if let Some(turn) = self.current.as_mut() {
                    turn.generation_started_at.get_or_insert(at);
                    turn.last_activity_at = at;
                }
                None
            }
            FrameInfo::GenerationFinished => {
                if self.at_learned(Learned::Generation, path)
                    && let Some(turn) = self.current.as_mut()
                {
                    turn.generation_finished_at = Some(at);
                    turn.last_activity_at = at;
                }
                None
            }
            FrameInfo::ToolCall {
                id,
                name,
                arguments_json,
            } => {
                if let Some(turn) = self.current.as_mut()
                    && !turn.tool_calls.iter().any(|c| c.id == id)
                {
                    turn.tool_calls.push(ToolCallRecord {
                        id,
                        name,
                        arguments_json,
                        result: None,
                        at_ms: ms(at),
                    });
                    turn.last_activity_at = at;
                }
                None
            }
            FrameInfo::ToolResult {
                tool_call_id,
                content,
                ..
            } => {
                if let Some(turn) = self.current.as_mut()
                    && let Some(call) = turn.tool_calls.iter_mut().find(|c| c.id == tool_call_id)
                {
                    call.result = Some(content);
                    turn.last_activity_at = at;
                }
                None
            }
            FrameInfo::ModelEvent { .. }
            | FrameInfo::Dispatch { .. }
            | FrameInfo::Custom { .. } => None,
        }
    }

    fn system(&mut self, _stage: &StageId, frame: SystemFrame, at: Duration) -> Option<TurnRecord> {
        match frame {
            // A barge-in originator (pipecrab-turn's BargeInStage) fires an
            // Interrupt on *every* speech onset, agent idle or not, so an
            // Interrupt only means "cut off" when the turn was engaged: a
            // generation still open, or agent audio at the tail within
            // [`SPEAKING_GRACE`]. An un-engaged Interrupt is the next
            // utterance's herald — the turn stays open and the following
            // `SpeechStarted` closes it as [`TurnEnd::NextTurn`]. Propagated
            // repeats find the turn already closed and no-op.
            SystemFrame::Interrupt => {
                let engaged = self.current.as_ref().is_some_and(|turn| {
                    let generating = turn.generation_started_at.is_some()
                        && turn.generation_finished_at.is_none();
                    let speaking = turn
                        .last_speech_at
                        .is_some_and(|last| at.saturating_sub(last) <= SPEAKING_GRACE);
                    generating || speaking
                });
                if engaged {
                    self.close(TurnEnd::Interrupted, at)
                } else {
                    None
                }
            }
            SystemFrame::Stop => self.close(TurnEnd::SessionEnd, at),
            SystemFrame::Error { message, .. } => {
                if let Some(turn) = self.current.as_mut()
                    && !turn.errors.contains(&message)
                {
                    turn.errors.push(message);
                }
                None
            }
            SystemFrame::Start => None,
        }
    }

    fn open(&mut self, origin: TurnOrigin, at: Duration) {
        self.current = Some(TurnState::new(origin, at, self.lost_total));
    }

    fn close(&mut self, end: TurnEnd, at: Duration) -> Option<TurnRecord> {
        let turn = self.current.take()?;
        let seq = self.turn_seq;
        self.turn_seq += 1;

        let interrupted = end == TurnEnd::Interrupted;
        let timings = TurnTimings {
            user_audio_secs: (turn.user_audio_secs > 0.0).then_some(turn.user_audio_secs),
            stt_ms: delta(turn.speech_stopped_at, turn.user_final_at),
            lm_ttft_ms: delta(turn.generation_started_at, turn.first_agent_partial_at),
            lm_total_ms: delta(turn.generation_started_at, turn.generation_finished_at),
            time_to_first_speech_ms: delta(turn.last_user_audio_at, turn.first_speech_at),
            response_latency_ms: delta(turn.speech_stopped_at, turn.first_speech_at),
            speech_before_interrupt_ms: interrupted
                .then(|| delta(turn.first_speech_at, Some(at)))
                .flatten(),
        };

        let mut stages: Vec<(Arc<str>, StageAgg)> = turn.stages.into_iter().collect();
        stages.sort_by_key(|(path, _)| {
            path.split('.')
                .map(|seg| seg.parse::<u64>().unwrap_or(u64::MAX))
                .collect::<Vec<_>>()
        });
        let stages = stages
            .into_iter()
            .map(|(path, agg)| StageTimings {
                name: agg.name.unwrap_or_else(|| path.clone()),
                path,
                frames: agg.frames,
                decide_ms: ms(agg.decide),
                decide_max_ms: ms(agg.decide_max),
                performs: agg.performs,
                perform_ms: ms(agg.perform),
                perform_max_ms: ms(agg.perform_max),
                errors: agg.errors,
                aborted: agg.aborted,
            })
            .collect();

        let agent_text = match turn.agent_finals.len() {
            0 => None,
            1 => Some(turn.agent_finals[0].clone()),
            _ => Some(Arc::from(turn.agent_finals.join(" "))),
        };

        Some(TurnRecord {
            session: self.session.clone(),
            turn: seq,
            origin: turn.origin,
            end,
            started_ms: ms(turn.started_at),
            ended_ms: ms(at),
            user_text: turn.user_text,
            agent_text,
            tool_calls: turn.tool_calls,
            interrupted_at_ms: interrupted.then(|| ms(at)),
            timings,
            stages,
            errors: turn.errors,
            lost_events: self.lost_total - turn.lost_at_open,
        })
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

fn delta(from: Option<Duration>, to: Option<Duration>) -> Option<f64> {
    match (from, to) {
        (Some(from), Some(to)) => Some(ms(to.saturating_sub(from))),
        _ => None,
    }
}
