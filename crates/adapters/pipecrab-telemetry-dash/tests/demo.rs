//! Manual demo: serve the dashboard with synthetic turns so it can be
//! eyeballed in a browser. Ignored by default; run with
//!
//! ```text
//! cargo test -p pipecrab-telemetry-dash --test demo -- --ignored --nocapture
//! ```
//!
//! and open the printed URL. The server stays up for two minutes, landing a
//! new turn every two seconds.

use std::sync::Arc;
use std::time::Duration;

use pipecrab_telemetry::{
    SessionInfo, StageTimings, TelemetrySink, ToolCallRecord, TurnEnd, TurnOrigin, TurnRecord,
    TurnTimings,
};
use pipecrab_telemetry_dash::DashboardSink;

fn stage(path: &str, name: &str, decide_ms: f64, perform_ms: f64) -> StageTimings {
    StageTimings {
        path: Arc::from(path),
        name: Arc::from(name),
        frames: 40,
        decide_ms,
        decide_max_ms: decide_ms / 4.0,
        performs: 12,
        perform_ms,
        perform_max_ms: perform_ms / 3.0,
        errors: 0,
        aborted: 0,
    }
}

/// Deterministic pseudo-noise so the demo needs no RNG dependency.
fn wobble(i: u64, scale: f64) -> f64 {
    (((i * 2_654_435_761) % 1_000) as f64 / 1_000.0 - 0.5) * scale
}

fn record(i: u64) -> TurnRecord {
    let interrupted = i % 7 == 3;
    let resp = 820.0 + wobble(i, 500.0);
    TurnRecord {
        session: SessionInfo {
            id: Arc::from("demo"),
            started_unix_ms: 1_754_000_000_000,
        },
        turn: i,
        origin: if i % 11 == 5 {
            TurnOrigin::Model
        } else {
            TurnOrigin::Speech
        },
        end: if interrupted {
            TurnEnd::Interrupted
        } else {
            TurnEnd::NextTurn
        },
        started_ms: i as f64 * 4_000.0,
        ended_ms: i as f64 * 4_000.0 + 3_000.0,
        user_text: Some(Arc::from(format!("what's on my calendar for day {i}?"))),
        agent_text: Some(Arc::from(if interrupted {
            "You have three meetings, the first—"
        } else {
            "You have three meetings; the first is at ten."
        })),
        tool_calls: if i.is_multiple_of(3) {
            vec![ToolCallRecord {
                id: Arc::from(format!("call-{i}")),
                name: Arc::from("calendar_lookup"),
                arguments_json: Arc::from(format!(r#"{{"day":{i}}}"#)),
                result: (!i.is_multiple_of(6)).then(|| Arc::from("3 events")),
                at_ms: i as f64 * 4_000.0 + 900.0,
            }]
        } else {
            Vec::new()
        },
        interrupted_at_ms: interrupted.then_some(i as f64 * 4_000.0 + 2_500.0),
        timings: TurnTimings {
            user_audio_secs: Some(1.4 + wobble(i, 0.8)),
            stt_ms: Some(105.0 + wobble(i, 60.0)),
            lm_ttft_ms: Some(190.0 + wobble(i, 120.0)),
            lm_total_ms: Some(880.0 + wobble(i, 300.0)),
            time_to_first_speech_ms: Some(resp + 480.0),
            response_latency_ms: Some(resp),
            speech_before_interrupt_ms: interrupted.then(|| 700.0 + wobble(i, 400.0)),
        },
        stages: vec![
            stage("0", "ResamplerStage<RubatoSincResampler>", 41.0, 0.0),
            stage("1", "VadStage<SherpaVad>", 66.0, 0.0),
            stage(
                "2",
                "SttStage<OfflineSherpaStt>",
                2.0,
                118.0 + wobble(i, 60.0),
            ),
            stage("3", "LmStage<LlamaCpp>", 3.0, 870.0 + wobble(i, 250.0)),
            stage("5", "TtsStage<SherpaTts>", 2.0, 460.0 + wobble(i, 160.0)),
        ],
        errors: Vec::new(),
        lost_events: 0,
    }
}

#[test]
#[ignore = "manual: serves synthetic data for eyeballing the dashboard"]
fn demo_server() {
    let mut sink = DashboardSink::serve("127.0.0.1:7878").expect("bind demo port");
    println!("dashboard: {}", sink.url());
    for i in 0..24 {
        sink.record(&record(i)).unwrap();
    }
    for i in 24..84 {
        std::thread::sleep(Duration::from_secs(2));
        sink.record(&record(i)).unwrap();
    }
}
