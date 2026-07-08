//! `VadStage` adapts a `VoiceActivityDetector` into a stage: it forwards each
//! `Audio` frame untouched (a tap) and emits `SpeechStarted` / `SpeechStopped`
//! system frames on the *edges* of speech, debounced by the hangover config.
//!
//! Deterministic and tokio-free (`block_on`), so it rides the default
//! `cargo test --workspace` path.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::executor::block_on;
use pipecrab_core::{AudioChunk, AudioFormat, DataFrame, Direction, SystemFrame};
use pipecrab_runtime::{PipelineBuilder, Received};
use pipecrab_vad::{VadConfig, VadError, VadStage, VadVerdict, VoiceActivityDetector};

/// A hardware-free detector that replays a scripted sequence of `is_speech`
/// verdicts, one per `detect` call, and rejects a wrong format — enough to
/// prove the edge logic without a model.
struct ScriptedVad {
    verdicts: Mutex<VecDeque<bool>>,
    format: AudioFormat,
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl VoiceActivityDetector for ScriptedVad {
    async fn detect(&self, _samples: &[f32], format: AudioFormat) -> Result<VadVerdict, VadError> {
        if format != self.format {
            return Err(VadError::UnsupportedFormat { expected: self.format, got: format });
        }
        let is_speech = self.verdicts.lock().unwrap().pop_front().unwrap_or(false);
        Ok(VadVerdict { speech_probability: if is_speech { 0.9 } else { 0.1 }, is_speech })
    }
}

#[test]
fn emits_edges_with_hangover_and_forwards_audio() {
    block_on(async {
        let fmt = AudioFormat::new(16_000, 1);
        // start after 2 speech windows, stop after 3 silence windows.
        let config = VadConfig { start_windows: 2, stop_windows: 3 };
        // Window-by-window verdicts. The lone `false` at index 4-5 is a brief
        // gap that the stop hangover (3) must ride out; the `true` at index 6
        // resets it, so speech only ends after the final run of three falses.
        let script = [
            false, // silence
            true, true, // 2 speech windows -> SpeechStarted
            false, false, // 2 silence < stop hangover of 3: no edge
            true, // back to speech: resets the stop run
            false, false, false, // 3 silence -> SpeechStopped
        ];
        let n_frames = script.len();
        let detector = ScriptedVad { verdicts: Mutex::new(script.into_iter().collect()), format: fmt };
        let (ends, driver) =
            PipelineBuilder::new().stage(VadStage::with_config(detector, config)).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            for _ in 0..n_frames {
                let chunk = AudioChunk::new(Arc::from(&[0.0f32; 2][..]), fmt);
                let _ = input.send_data(DataFrame::Audio(chunk)).await;
            }
            // Returning drops `input`, cascading shutdown through the pipeline.
        };

        let drain = async move {
            let mut edges = Vec::new();
            let mut audio_forwarded = 0;
            while let Some(received) = output.recv().await {
                match received {
                    Received::Data(DataFrame::Audio(_)) => audio_forwarded += 1,
                    Received::Sys(Direction::Down, SystemFrame::SpeechStarted) => {
                        edges.push("started")
                    }
                    Received::Sys(Direction::Down, SystemFrame::SpeechStopped) => {
                        edges.push("stopped")
                    }
                    _ => {}
                }
            }
            (edges, audio_forwarded)
        };

        let (_, (edges, audio_forwarded), _) = futures::join!(feed, drain, driver);
        assert_eq!(edges, vec!["started", "stopped"], "one clean start/stop pair");
        assert_eq!(audio_forwarded, n_frames, "every audio frame is forwarded (VAD is a tap)");
    });
}

#[test]
fn wrong_format_surfaces_a_recoverable_error() {
    block_on(async {
        let detector = ScriptedVad {
            verdicts: Mutex::new(VecDeque::new()),
            format: AudioFormat::new(16_000, 1),
        };
        let (ends, driver) = PipelineBuilder::new().stage(VadStage::new(detector)).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            // 48 kHz stereo — not the 16 kHz mono the detector accepts.
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
        assert!(!fatal, "a detection failure is recoverable, not fatal");
        assert!(message.contains("format mismatch"), "unexpected message: {message}");
    });
}
