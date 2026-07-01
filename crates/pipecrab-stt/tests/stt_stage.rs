//! `SttStage` adapts a `Transcriber` into a stage: an `Audio` frame in, a
//! `Transcript` frame out, and a rejected format surfaced as an `Error`.
//!
//! Deterministic and tokio-free (`block_on`), so it rides the default
//! `cargo test --workspace` path — the browser-inference risks are retired
//! separately by `examples/stt-web`.

use std::sync::Arc;

use async_trait::async_trait;
use futures::executor::block_on;
use pipecrab_core::{AudioChunk, AudioFormat, DataFrame, Direction, SystemFrame};
use pipecrab_runtime::{PipelineBuilder, Received};
use pipecrab_stt::{SttError, SttStage, Transcriber};

/// A hardware-free transcriber: it reports the sample count it was handed and
/// accepts only its configured format — enough to prove the seam and its format
/// contract without loading a model.
struct MockTranscriber {
    format: AudioFormat,
}

#[async_trait(?Send)]
impl Transcriber for MockTranscriber {
    async fn transcribe(&self, samples: &[f32], format: AudioFormat) -> Result<String, SttError> {
        if format != self.format {
            return Err(SttError::UnsupportedFormat { expected: self.format, got: format });
        }
        Ok(format!("heard {} samples", samples.len()))
    }
}

#[test]
fn audio_frame_becomes_transcript() {
    block_on(async {
        let fmt = AudioFormat::new(16_000, 1);
        let stage = SttStage::new(MockTranscriber { format: fmt });
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let chunk = AudioChunk::new(Arc::from(&[0.0f32; 3][..]), fmt);
            let _ = input.send_data(DataFrame::Audio(chunk)).await;
            // Returning drops `input`, cascading shutdown through the pipeline.
        };

        let drain = async move {
            let mut transcript = None;
            while let Some(received) = output.recv().await {
                if let Received::Data(DataFrame::Transcript(text)) = received {
                    transcript = Some(text);
                }
            }
            transcript
        };

        let (_, transcript, _) = futures::join!(feed, drain, driver);
        assert_eq!(transcript.as_deref(), Some("heard 3 samples"));
    });
}

#[test]
fn wrong_format_surfaces_a_recoverable_error() {
    block_on(async {
        let stage = SttStage::new(MockTranscriber { format: AudioFormat::new(16_000, 1) });
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            // 48 kHz stereo — not the 16 kHz mono the transcriber accepts.
            let chunk = AudioChunk::new(Arc::from(&[0.0f32; 4][..]), AudioFormat::new(48_000, 2));
            let _ = input.send_data(DataFrame::Audio(chunk)).await;
        };

        let drain = async move {
            let mut error = None;
            while let Some(received) = output.recv().await {
                if let Received::Sys(Direction::Up, SystemFrame::Error { message, fatal }) = received {
                    error = Some((message, fatal));
                }
            }
            error
        };

        let (_, error, _) = futures::join!(feed, drain, driver);
        let (message, fatal) = error.expect("a format mismatch should surface an Error frame");
        assert!(!fatal, "a transcription failure is recoverable, not fatal");
        assert!(message.contains("format mismatch"), "unexpected message: {message}");
    });
}
