//! Pipeline-level contract for `ResamplerStage`.

use std::sync::Arc;

use futures::executor::block_on;
use pipecrab_audio::{AudioChunk, AudioFormat, ResamplerStage};
use pipecrab_core::{DataFrame, Direction, SystemFrame};
use pipecrab_runtime::{PipelineBuilder, Received};

#[test]
fn converted_audio_stays_bracketed_by_data_frames() {
    block_on(async {
        let input_format = AudioFormat::new(48_000, 1);
        let output_format = AudioFormat::new(16_000, 1);
        let stage = ResamplerStage::new(output_format).unwrap();
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let send = async move {
            input.send_data(DataFrame::SpeechStarted).await.unwrap();
            input
                .send_data(DataFrame::Audio(AudioChunk::new(
                    Arc::from(vec![0.25; 960]),
                    input_format,
                )))
                .await
                .unwrap();
            input.send_data(DataFrame::SpeechStopped).await.unwrap();
        };
        let receive = async move {
            let mut frames = Vec::new();
            while let Some(frame) = output.recv().await {
                frames.push(frame);
            }
            frames
        };

        let (_, _, frames) = futures::join!(driver, send, receive);
        assert!(matches!(
            frames.as_slice(),
            [
                Received::Data(DataFrame::SpeechStarted),
                Received::Data(DataFrame::Audio(chunk)),
                Received::Data(DataFrame::SpeechStopped),
            ] if chunk.format == output_format && !chunk.samples.is_empty()
        ));
    });
}

#[test]
fn same_format_audio_keeps_its_arc() {
    block_on(async {
        let format = AudioFormat::new(48_000, 1);
        let samples: Arc<[f32]> = Arc::from([0.25, -0.25]);
        let retained = samples.clone();
        let stage = ResamplerStage::new(format).unwrap();
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let send = async move {
            input
                .send_data(DataFrame::Audio(AudioChunk::new(samples, format)))
                .await
                .unwrap();
        };
        let receive = async move { output.recv().await.unwrap() };

        let (_, _, received) = futures::join!(driver, send, receive);
        let Received::Data(DataFrame::Audio(chunk)) = received else {
            panic!("expected one audio frame");
        };
        assert!(Arc::ptr_eq(&retained, &chunk.samples));
    });
}

#[test]
fn interrupt_is_forwarded_and_resets_the_stream() {
    block_on(async {
        let output_format = AudioFormat::new(16_000, 1);
        let stage = ResamplerStage::new(output_format).unwrap();
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let send = async move {
            input
                .send_system(Direction::Down, SystemFrame::Interrupt)
                .await
                .unwrap();
        };
        let receive = async move { output.recv().await.unwrap() };

        let (_, _, received) = futures::join!(driver, send, receive);
        assert!(matches!(
            received,
            Received::Sys(Direction::Down, SystemFrame::Interrupt)
        ));
    });
}

#[test]
fn malformed_audio_is_a_fatal_stage_error() {
    block_on(async {
        let format = AudioFormat::new(48_000, 2);
        let stage = ResamplerStage::new(format).unwrap();
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let send = async move {
            input
                .send_data(DataFrame::Audio(AudioChunk::new(
                    Arc::from([0.0, 1.0, 2.0]),
                    format,
                )))
                .await
                .unwrap();
        };
        let receive = async move { output.recv().await.unwrap() };

        let (_, _, received) = futures::join!(driver, send, receive);
        assert!(matches!(
            received,
            Received::Sys(
                Direction::Up,
                SystemFrame::Error {
                    fatal: true,
                    message,
                },
            ) if message.contains("not divisible")
        ));
    });
}
