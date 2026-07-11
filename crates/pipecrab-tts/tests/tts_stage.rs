//! `TtsStage` adapts a `Synthesizer` into a stage: a final agent `Transcript`
//! in, a stream of `Audio` frames out, and a barge-in that stops emission within
//! one chunk while cancelling the engine.
//!
//! Deterministic and tokio-free (`block_on`), so it rides the default
//! `cargo test --workspace` path — real browser/native synthesis is exercised
//! separately by the engine crates.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::executor::block_on;
use futures::sink::SinkExt;
use futures::stream::StreamExt;
use pipecrab_core::{AudioChunk, AudioFormat, DataFrame, Direction, SystemFrame, Transcript};
use pipecrab_runtime::{PipelineBuilder, Received};
use pipecrab_tts::{Synthesizer, TtsAudioStream, TtsError};

/// The one chunk our mock emits per synthesis; a distinct sample value makes it
/// identifiable at the output.
fn chunk() -> AudioChunk {
    AudioChunk::new(Arc::from(&[0.25f32][..]), AudioFormat::new(24_000, 1))
}

// --- A synthesizer whose stream emits one chunk, then parks. -----------------

/// Emits a single [`AudioChunk`], then signals `emitted` and parks on `block`
/// forever. The park models an engine still producing when the barge-in lands:
/// the run loop must drop the in-flight `perform` (dropping the stream, so
/// `block`'s sender is cancelled) rather than wait for a second chunk. `cancel`
/// flips `cancelled`, proving the control call reached the engine.
struct ParkingSynth {
    emitted: mpsc::Sender<()>,
    block: Mutex<Option<oneshot::Receiver<()>>>,
    cancelled: Arc<AtomicBool>,
}

/// One `unfold` step of the parking stream.
enum Gen {
    First { emitted: mpsc::Sender<()>, block: oneshot::Receiver<()> },
    Park { emitted: mpsc::Sender<()>, block: oneshot::Receiver<()> },
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Synthesizer for ParkingSynth {
    fn output_format(&self) -> AudioFormat {
        AudioFormat::new(24_000, 1)
    }

    async fn synthesize(&self, _text: &str) -> Result<TtsAudioStream, TtsError> {
        let emitted = self.emitted.clone();
        let block = self.block.lock().unwrap().take().expect("synthesize runs once");
        let stream = futures::stream::unfold(Gen::First { emitted, block }, |state| async move {
            match state {
                Gen::First { emitted, block } => {
                    // Yield exactly one chunk, then move to the parked state.
                    Some((Ok(chunk()), Gen::Park { emitted, block }))
                }
                Gen::Park { mut emitted, block } => {
                    // One chunk is out; tell the test, then park. A barge-in
                    // drops this future (cancelling `block`) before it resolves.
                    let _ = emitted.send(()).await;
                    let _ = block.await;
                    None
                }
            }
        });
        Ok(stream.boxed())
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

#[test]
fn barge_in_stops_emission_within_one_chunk() {
    block_on(async {
        let cancelled = Arc::new(AtomicBool::new(false));
        let (emitted_tx, mut emitted_rx) = mpsc::channel::<()>(1);
        let (block_tx, block_rx) = oneshot::channel::<()>();

        let synth = ParkingSynth {
            emitted: emitted_tx,
            block: Mutex::new(Some(block_rx)),
            cancelled: cancelled.clone(),
        };
        let (ends, driver) = PipelineBuilder::new().stage(pipecrab_tts::TtsStage::new(synth)).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let _ = input.send_data(Transcript::agent_final("hello there").into()).await;
            // Wait until the first chunk is out and the engine has parked, then
            // barge in.
            emitted_rx.next().await.expect("the synthesizer must emit one chunk");
            let _ = input.send_system(Direction::Down, SystemFrame::Interrupt).await;
            // Returning drops `input`, cascading shutdown through the pipeline.
        };

        let drain = async move {
            let mut audio = 0usize;
            while let Some(received) = output.recv().await {
                if let Received::Data(DataFrame::Audio(_)) = received {
                    audio += 1;
                }
            }
            audio
        };

        let (_, audio, _) = futures::join!(feed, drain, driver);
        assert_eq!(audio, 1, "emission must stop within one chunk of barge-in");
        assert!(cancelled.load(Ordering::SeqCst), "barge-in must reach the engine's cancel()");
        assert!(
            block_tx.is_canceled(),
            "the in-flight perform must have been dropped (its parked receiver gone)",
        );
    });
}

// --- A synthesizer that streams a fixed number of chunks and completes. -------

/// Yields `n` chunks then ends — enough to prove a final agent transcript is
/// turned into a run of `Audio` frames, and that non-agent-final frames forward.
struct FixedSynth {
    n: usize,
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Synthesizer for FixedSynth {
    fn output_format(&self) -> AudioFormat {
        AudioFormat::new(24_000, 1)
    }

    async fn synthesize(&self, _text: &str) -> Result<TtsAudioStream, TtsError> {
        let items: Vec<Result<AudioChunk, TtsError>> = (0..self.n).map(|_| Ok(chunk())).collect();
        Ok(futures::stream::iter(items).boxed())
    }

    fn cancel(&self) {}
}

#[test]
fn final_agent_transcript_becomes_audio_stream() {
    block_on(async {
        let (ends, driver) =
            PipelineBuilder::new().stage(pipecrab_tts::TtsStage::new(FixedSynth { n: 3 })).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            // A user transcript is not the agent's speech: it must forward.
            let _ = input.send_data(Transcript::user_final("hi").into()).await;
            // An agent partial is not final: it must forward, not synthesize.
            let _ = input.send_data(Transcript::agent_partial("typ").into()).await;
            // The agent's final speech: synthesized into audio.
            let _ = input.send_data(Transcript::agent_final("done").into()).await;
        };

        let drain = async move {
            let mut audio = 0usize;
            let mut forwarded_transcripts = 0usize;
            while let Some(received) = output.recv().await {
                match received {
                    Received::Data(DataFrame::Audio(_)) => audio += 1,
                    Received::Data(DataFrame::Transcript(_)) => forwarded_transcripts += 1,
                    _ => {}
                }
            }
            (audio, forwarded_transcripts)
        };

        let (_, (audio, forwarded), _) = futures::join!(feed, drain, driver);
        assert_eq!(audio, 3, "the final agent transcript yields one Audio frame per chunk");
        assert_eq!(forwarded, 2, "user-final and agent-partial forward untouched");
    });
}
