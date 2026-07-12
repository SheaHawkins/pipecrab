//! Verifies that typed audio frames pass through a pipeline unchanged.

use futures::executor::block_on;
use pipecrab_audio::mock::{MockSink, MockSource};
use pipecrab_audio::{AudioFormat, AudioSink, AudioSource};
use pipecrab_core::{DataFrame, Direction, Processor, SystemFrame};
use pipecrab_runtime::{maybe_async_trait, Outbound, PipelineBuilder, Received, Stage, StageError};

/// Uses the [`Processor`] defaults to forward every frame.
struct EchoStage;

impl Processor for EchoStage {
    type Effect = ();
}

maybe_async_trait! {
    impl Stage for EchoStage {
        async fn perform(&self, _effect: (), _out: &Outbound) -> Result<(), StageError> {
            Ok(())
        }
    }
}

#[test]
fn echo_passthrough_preserves_ramp() {
    let format = AudioFormat::new(48_000, 1);
    let chunk_frames = 960; // ~20 ms @ 48 kHz, the real capture chunk size.
    let chunks = 5;
    let total = (chunk_frames * chunks) as u32;

    let source = MockSource::ramp(format, chunk_frames, chunks);
    let sink = MockSink::new(format);

    let (ends, driver) = PipelineBuilder::new().stage(EchoStage).build().start();
    let input = ends.input;
    let output = ends.output;

    // In-pump: Start at boot, then each captured chunk as a typed Audio frame.
    // Dropping `input` at scope end closes the head and cascades shutdown.
    let pump_in = async move {
        let mut source = source;
        input
            .send_system(Direction::Down, SystemFrame::Start)
            .await
            .unwrap();
        while let Ok(Some(chunk)) = source.next_chunk().await {
            input.send_data(DataFrame::Audio(chunk)).await.unwrap();
        }
    };

    // Out-pump: play Audio frames into the sink; ignore the forwarded Start.
    // Exhaustive match, no downcast. Returns the sink so the test can inspect it.
    let pump_out = async move {
        let mut sink = sink;
        let mut output = output;
        while let Some(received) = output.recv().await {
            match received {
                Received::Data(DataFrame::Audio(chunk)) => sink.play(chunk).await.unwrap(),
                Received::Data(_) => {}
                Received::Sys(_, _) => {}
            }
        }
        sink
    };

    let (_, _, sink) = block_on(async { futures::join!(driver, pump_in, pump_out) });

    let expected: Vec<f32> = (0..total).map(|i| i as f32).collect();
    assert_eq!(
        sink.samples(),
        expected,
        "passthrough must preserve the ramp exactly"
    );
    assert_eq!(
        sink.chunks().len(),
        chunks,
        "each chunk arrives as its own frame"
    );
}
