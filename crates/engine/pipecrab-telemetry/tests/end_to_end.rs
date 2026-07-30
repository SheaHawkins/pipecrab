//! The whole chain, natively: a real pipeline reporting into a [`Session`]'s
//! hub, the worker thread assembling turns, a sink collecting the records.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::{Arc, Mutex};

use futures::executor::block_on;
use futures::future::join;
use pipecrab_core::{
    AudioChunk, AudioFormat, DataFrame, ModelFrame, Processor, ToolCall, Transcript,
};
use pipecrab_runtime::{Outbound, PipelineBuilder, Stage, StageError};
use pipecrab_telemetry::{SinkError, Telemetry, TelemetrySink, TurnEnd, TurnOrigin, TurnRecord};

/// Collects records into shared memory for assertions.
#[derive(Clone, Default)]
struct Collect(Arc<Mutex<Vec<TurnRecord>>>);

impl TelemetrySink for Collect {
    fn record(&mut self, turn: &TurnRecord) -> Result<(), SinkError> {
        self.0.lock().unwrap().push(turn.clone());
        Ok(())
    }
}

struct PassThrough;
impl Processor for PassThrough {
    type Effect = ();
    fn name(&self) -> &'static str {
        "pass"
    }
}
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl Stage for PassThrough {
    async fn perform(&self, _effect: (), _out: &Outbound) -> Result<(), StageError> {
        Ok(())
    }
}

fn audio(samples: usize) -> DataFrame {
    DataFrame::Audio(AudioChunk::new(
        Arc::from(vec![0.0f32; samples].as_slice()),
        AudioFormat::new(16_000, 1),
    ))
}

#[test]
fn a_fed_turn_comes_out_of_the_sink_as_one_record() {
    let sink = Collect::default();
    let session = Telemetry::builder()
        .sink(sink.clone())
        .build()
        .session_with_id("e2e");

    let (ends, driver) = PipelineBuilder::new()
        .stage(PassThrough)
        .observer(session.observer())
        .build()
        .start();
    let input = ends.input;
    let _output = ends.output;

    let feeder = async move {
        // One spoken turn, as the frames downstream of a VAD gate + STT + LM
        // would carry it.
        input.send_data(DataFrame::SpeechStarted).await.unwrap();
        input.send_data(audio(16_000)).await.unwrap(); // 1 s utterance
        input.send_data(DataFrame::SpeechStopped).await.unwrap();
        input
            .send_data(Transcript::user_final("ping").into())
            .await
            .unwrap();
        input
            .send_data(ModelFrame::GenerationStarted.into())
            .await
            .unwrap();
        input
            .send_data(
                ToolCall {
                    id: Arc::from("call-1"),
                    name: Arc::from("echo"),
                    arguments_json: Arc::from("{}"),
                }
                .into(),
            )
            .await
            .unwrap();
        input
            .send_data(Transcript::agent_final("pong").into())
            .await
            .unwrap();
        input
            .send_data(ModelFrame::GenerationFinished.into())
            .await
            .unwrap();
        input.send_data(audio(24_000)).await.unwrap(); // agent speech
    };
    block_on(async { join(feeder, driver).await });
    session.finish();

    let records = sink.0.lock().unwrap();
    assert_eq!(records.len(), 1, "one turn, one record");
    let turn = &records[0];
    assert_eq!(&*turn.session.id, "e2e");
    assert_eq!(turn.origin, TurnOrigin::Speech);
    assert_eq!(turn.end, TurnEnd::SessionEnd);
    assert_eq!(turn.user_text.as_deref(), Some("ping"));
    assert_eq!(turn.agent_text.as_deref(), Some("pong"));
    assert_eq!(turn.tool_calls.len(), 1);
    assert_eq!(&*turn.tool_calls[0].name, "echo");
    assert!((turn.timings.user_audio_secs.unwrap() - 1.0).abs() < 1e-9);
    assert!(
        turn.timings.time_to_first_speech_ms.is_some(),
        "agent audio at the tail must stamp first speech"
    );
    assert!(
        turn.stages.iter().any(|s| &*s.name == "pass"),
        "the observed stage must appear in the turn's spans: {:?}",
        turn.stages
    );
    assert_eq!(turn.lost_events, 0);
}
