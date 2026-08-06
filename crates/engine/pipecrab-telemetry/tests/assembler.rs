//! [`TurnAssembler`] behavior over synthetic event streams: a full voice
//! turn with timings, propagation dedupe, tool-call correlation, barge-in,
//! and model-initiated turns.

use std::sync::Arc;
use std::time::Duration;

use pipecrab_core::{Direction, Disposition, Finality, Role, SystemFrame};
use pipecrab_runtime::{StageId, TAIL_STAGE};
use pipecrab_telemetry::{
    Event, EventKind, FrameInfo, SessionInfo, TurnAssembler, TurnEnd, TurnOrigin, TurnRecord,
};

fn stage(path: &str, name: &'static str) -> StageId {
    StageId {
        path: Arc::from(path),
        name,
    }
}

fn at(ms: u64) -> Duration {
    Duration::from_millis(ms)
}

fn ev(ms: u64, stage_id: Option<StageId>, kind: EventKind) -> Event {
    Event {
        at: at(ms),
        stage: stage_id,
        kind,
    }
}

fn frame(ms: u64, path: &str, name: &'static str, info: FrameInfo) -> Event {
    ev(ms, Some(stage(path, name)), EventKind::FrameIn(info))
}

fn assembler() -> TurnAssembler {
    TurnAssembler::new(SessionInfo {
        id: Arc::from("s1"),
        started_unix_ms: 1_000,
    })
}

/// The wired report for a vad → stt → lm → tts → tail pipeline.
fn wired() -> Event {
    ev(
        0,
        None,
        EventKind::Wired {
            stages: vec![
                stage("0", "vad"),
                stage("1", "stt"),
                stage("2", "lm"),
                stage("3", "tts"),
                stage("4", TAIL_STAGE),
            ],
        },
    )
}

fn user_final(text: &str) -> FrameInfo {
    FrameInfo::Transcript {
        text: Some(Arc::from(text)),
        len: text.len(),
        role: Role::User,
        finality: Finality::Final,
    }
}

fn agent_partial(len: usize) -> FrameInfo {
    FrameInfo::Transcript {
        text: None,
        len,
        role: Role::Agent,
        finality: Finality::Partial { stable: len },
    }
}

fn agent_final(text: &str) -> FrameInfo {
    FrameInfo::Transcript {
        text: Some(Arc::from(text)),
        len: text.len(),
        role: Role::Agent,
        finality: Finality::Final,
    }
}

fn audio(samples: usize) -> FrameInfo {
    FrameInfo::Audio {
        samples,
        sample_rate: 16_000,
        channels: 1,
    }
}

/// One complete spoken turn; edges and frames propagate across boundaries the
/// way a real pipeline duplicates them.
fn full_turn(assembler: &mut TurnAssembler) -> Vec<TurnRecord> {
    let mut closed = Vec::new();
    let mut push = |e: Event| {
        if let Some(record) = assembler.push(e) {
            closed.push(record);
        }
    };

    push(wired());
    // Utterance: edge boundary is "1" (VAD's successor); duplicates at "2".
    push(frame(100, "1", "stt", FrameInfo::SpeechStarted));
    push(frame(101, "2", "lm", FrameInfo::SpeechStarted)); // propagation dup
    push(frame(110, "1", "stt", audio(16_000))); // 1.0 s of utterance
    push(frame(590, "1", "stt", audio(8_000))); // +0.5 s, last chunk at 590
    push(frame(600, "1", "stt", FrameInfo::SpeechStopped));
    push(frame(601, "2", "lm", FrameInfo::SpeechStopped)); // dup
    // STT final lands at LM ingress at 700 (and propagates).
    push(frame(700, "2", "lm", user_final("what's the weather")));
    push(frame(701, "3", "tts", user_final("what's the weather"))); // dup
    // Generation with one tool call.
    push(frame(720, "3", "tts", FrameInfo::GenerationStarted));
    push(frame(900, "3", "tts", agent_partial(4)));
    push(frame(
        950,
        "3",
        "tts",
        FrameInfo::ToolCall {
            id: Arc::from("call-1"),
            name: Arc::from("weather"),
            arguments_json: Arc::from(r#"{"city":"paris"}"#),
        },
    ));
    push(frame(
        951,
        "4",
        TAIL_STAGE,
        FrameInfo::ToolCall {
            id: Arc::from("call-1"),
            name: Arc::from("weather"),
            arguments_json: Arc::from(r#"{"city":"paris"}"#),
        },
    )); // dup by id
    push(frame(
        1_200,
        "2",
        "lm",
        FrameInfo::ToolResult {
            tool_call_id: Arc::from("call-1"),
            name: Arc::from("weather"),
            content: Arc::from("sunny, 18C"),
            respond: true,
        },
    ));
    push(frame(1_210, "3", "tts", FrameInfo::GenerationStarted)); // 2nd gen
    push(frame(
        1_400,
        "3",
        "tts",
        agent_final("It's sunny in Paris."),
    ));
    push(frame(1_450, "3", "tts", FrameInfo::GenerationFinished));
    // First agent audio reaches the tail at 1500.
    push(frame(1_500, "4", TAIL_STAGE, audio(24_000)));
    push(frame(1_600, "4", TAIL_STAGE, audio(24_000)));
    closed
}

#[test]
fn assembles_a_full_turn_with_timings_and_tool_calls() {
    let mut asm = assembler();
    let closed = full_turn(&mut asm);
    assert!(closed.is_empty(), "turn should still be open");
    let turn = asm.finish().expect("open turn flushed at session end");

    assert_eq!(turn.turn, 0);
    assert_eq!(turn.origin, TurnOrigin::Speech);
    assert_eq!(turn.end, TurnEnd::SessionEnd);
    assert_eq!(turn.user_text.as_deref(), Some("what's the weather"));
    assert_eq!(turn.agent_text.as_deref(), Some("It's sunny in Paris."));

    // Tool call deduped by id and correlated with its result.
    assert_eq!(turn.tool_calls.len(), 1);
    let call = &turn.tool_calls[0];
    assert_eq!(&*call.name, "weather");
    assert_eq!(call.result.as_deref(), Some("sunny, 18C"));

    let t = &turn.timings;
    // 1.5 s of utterance audio, sample-counted.
    assert!((t.user_audio_secs.unwrap() - 1.5).abs() < 1e-9);
    // SpeechStopped(600) -> user final (700).
    assert!((t.stt_ms.unwrap() - 100.0).abs() < 1e-9);
    // First GenerationStarted(720) -> first agent partial (900).
    assert!((t.lm_ttft_ms.unwrap() - 180.0).abs() < 1e-9);
    // First GenerationStarted(720) -> last GenerationFinished(1450).
    assert!((t.lm_total_ms.unwrap() - 730.0).abs() < 1e-9);
    // Last utterance chunk (590) -> first tail audio (1500).
    assert!((t.time_to_first_speech_ms.unwrap() - 910.0).abs() < 1e-9);
    // SpeechStopped(600) -> first tail audio (1500).
    assert!((t.response_latency_ms.unwrap() - 900.0).abs() < 1e-9);
}

#[test]
fn next_speech_started_closes_the_turn_and_opens_the_next() {
    let mut asm = assembler();
    let _ = full_turn(&mut asm);
    let closed = asm.push(frame(2_000, "1", "stt", FrameInfo::SpeechStarted));
    let turn = closed.expect("previous turn closes when the next opens");
    assert_eq!(turn.turn, 0);
    assert_eq!(turn.end, TurnEnd::NextTurn);

    // The dup at another boundary must not close the new turn.
    assert!(
        asm.push(frame(2_001, "2", "lm", FrameInfo::SpeechStarted))
            .is_none()
    );
    let second = asm.finish().expect("second turn open");
    assert_eq!(second.turn, 1);
}

fn interrupt(ms: u64, path: &str, name: &'static str) -> Event {
    ev(
        ms,
        Some(stage(path, name)),
        EventKind::SysIn {
            dir: Direction::Down,
            frame: SystemFrame::Interrupt,
        },
    )
}

#[test]
fn interrupt_while_speaking_closes_the_turn_as_interrupted_once() {
    let mut asm = assembler();
    let _ = full_turn(&mut asm);
    // Last tail audio was at 1600; 1700 is within the speaking grace.
    let turn = asm
        .push(interrupt(1_700, "0", "vad"))
        .expect("interrupt during agent speech closes the open turn");
    assert_eq!(turn.end, TurnEnd::Interrupted);
    assert!((turn.interrupted_at_ms.unwrap() - 1_700.0).abs() < 1e-9);
    // First agent audio (1500) -> interrupt (1700).
    assert!((turn.timings.speech_before_interrupt_ms.unwrap() - 200.0).abs() < 1e-9);
    // The interrupt propagating to later stages closes nothing further.
    assert!(asm.push(interrupt(1_701, "1", "stt")).is_none());
    assert!(asm.finish().is_none());
}

#[test]
fn interrupt_mid_generation_closes_the_turn_as_interrupted() {
    let mut asm = assembler();
    asm.push(wired());
    asm.push(frame(100, "1", "stt", FrameInfo::SpeechStarted));
    asm.push(frame(700, "2", "lm", user_final("hi")));
    asm.push(frame(720, "3", "tts", FrameInfo::GenerationStarted));
    // No agent audio yet — the open generation alone makes the turn engaged.
    let turn = asm
        .push(interrupt(800, "0", "vad"))
        .expect("interrupt mid-generation closes the open turn");
    assert_eq!(turn.end, TurnEnd::Interrupted);
    // Never reached speech, so there is no speech_before_interrupt.
    assert!(turn.timings.speech_before_interrupt_ms.is_none());
}

#[test]
fn idle_interrupt_is_the_next_utterances_herald_and_leaves_the_turn_open() {
    let mut asm = assembler();
    let _ = full_turn(&mut asm);
    // A barge-in originator fires on every onset; 5s after the agent's last
    // audio this is not a cut-off. The turn must stay open...
    assert!(asm.push(interrupt(6_600, "0", "vad")).is_none());
    // ...and the following SpeechStarted closes it as an ordinary next turn.
    let turn = asm
        .push(frame(6_650, "1", "stt", FrameInfo::SpeechStarted))
        .expect("next SpeechStarted closes the previous turn");
    assert_eq!(turn.end, TurnEnd::NextTurn);
    assert_eq!(turn.interrupted_at_ms, None);
}

#[test]
fn tick_quiet_closes_a_finished_turn_but_not_an_active_one() {
    let mut asm = assembler();
    let _ = full_turn(&mut asm);
    // Last activity is the tail audio at 1600. Too soon: nothing closes.
    assert!(
        asm.tick(at(2_000)).is_none(),
        "still inside the quiet window"
    );
    // Quiet long enough: the turn closes as completed, ended at last activity.
    let turn = asm.tick(at(3_500)).expect("quiet turn closes");
    assert_eq!(turn.end, TurnEnd::Completed);
    assert!((turn.ended_ms - 1_600.0).abs() < 1e-9);
    assert!(asm.finish().is_none(), "nothing left open");

    // A turn whose generation has not finished never quiet-closes.
    let mut asm = assembler();
    asm.push(wired());
    asm.push(frame(100, "1", "stt", FrameInfo::SpeechStarted));
    asm.push(frame(700, "2", "lm", user_final("hi")));
    asm.push(frame(720, "3", "tts", FrameInfo::GenerationStarted));
    assert!(asm.tick(at(60_000)).is_none(), "open generation stays open");
}

#[test]
fn generation_without_speech_opens_a_model_turn() {
    let mut asm = assembler();
    asm.push(wired());
    asm.push(frame(500, "3", "tts", FrameInfo::GenerationStarted));
    asm.push(frame(600, "3", "tts", agent_final("Your build finished.")));
    asm.push(frame(650, "3", "tts", FrameInfo::GenerationFinished));
    asm.push(frame(700, "4", TAIL_STAGE, audio(24_000)));
    let turn = asm.finish().expect("model turn flushed");
    assert_eq!(turn.origin, TurnOrigin::Model);
    assert_eq!(turn.user_text, None);
    assert_eq!(turn.agent_text.as_deref(), Some("Your build finished."));
    // No utterance to measure from, but the generation window exists.
    assert!(turn.timings.time_to_first_speech_ms.is_none());
    assert!((turn.timings.lm_total_ms.unwrap() - 150.0).abs() < 1e-9);
}

#[test]
fn stage_timings_aggregate_decide_and_perform_pairs() {
    let mut asm = assembler();
    asm.push(wired());
    asm.push(frame(100, "1", "stt", FrameInfo::SpeechStarted)); // opens turn
    // One decide pair (10 ms) and one perform pair (200 ms, error) on "2".
    asm.push(frame(200, "2", "lm", user_final("hi")));
    asm.push(ev(
        210,
        Some(stage("2", "lm")),
        EventKind::Decided {
            disposition: Disposition::Drop,
            effects: 1,
        },
    ));
    asm.push(ev(300, Some(stage("2", "lm")), EventKind::PerformStart));
    asm.push(ev(
        500,
        Some(stage("2", "lm")),
        EventKind::PerformEnd {
            outcome: pipecrab_runtime::PerformOutcome::Err {
                message: Arc::from("model exploded"),
                fatal: false,
            },
        },
    ));
    let turn = asm.finish().expect("turn flushed");
    let lm = turn
        .stages
        .iter()
        .find(|s| &*s.path == "2")
        .expect("lm stage aggregated");
    assert_eq!(lm.frames, 1);
    assert!((lm.decide_ms - 10.0).abs() < 1e-9);
    assert_eq!(lm.performs, 1);
    assert!((lm.perform_ms - 200.0).abs() < 1e-9);
    assert_eq!(lm.errors, 1);
}

#[test]
fn lost_events_are_attributed_to_the_open_turn() {
    let mut asm = assembler();
    asm.push(wired());
    asm.push(frame(100, "1", "stt", FrameInfo::SpeechStarted));
    asm.push(ev(150, None, EventKind::Lost { count: 7 }));
    let turn = asm.finish().expect("turn flushed");
    assert_eq!(turn.lost_events, 7);
}
