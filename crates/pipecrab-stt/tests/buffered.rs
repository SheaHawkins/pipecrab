//! `Buffered` adapts a one-shot `Transcriber` to the `StreamingTranscriber`
//! protocol: it accumulates `feed`s (emitting no partials) and produces a single
//! `Final` on `end_utterance`. These tests pin that behavior plus the two honest
//! limitations — deferred format rejection and stale-result discard on cancel.
//!
//! Deterministic and tokio-free (`block_on`), so it rides the default
//! `cargo test --workspace` path.

use std::sync::Mutex;

use async_trait::async_trait;
use futures::channel::oneshot;
use futures::executor::block_on;
use pipecrab_core::AudioFormat;
use pipecrab_stt::{Buffered, SttError, SttEvent, StreamingTranscriber, Transcriber};

/// A hardware-free one-shot transcriber: reports the sample count it was handed
/// and accepts only its configured format.
struct CountingTranscriber {
    format: AudioFormat,
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Transcriber for CountingTranscriber {
    async fn transcribe(&self, samples: &[f32], format: AudioFormat) -> Result<String, SttError> {
        if format != self.format {
            return Err(SttError::UnsupportedFormat { expected: self.format, got: format });
        }
        Ok(format!("heard {} samples", samples.len()))
    }
}

/// A one-shot transcriber whose `transcribe` blocks until a oneshot fires — used
/// to hold an `end_utterance` in flight while a `cancel` races in.
struct GatedTranscriber {
    gate: Mutex<Option<oneshot::Receiver<()>>>,
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Transcriber for GatedTranscriber {
    async fn transcribe(&self, _samples: &[f32], _format: AudioFormat) -> Result<String, SttError> {
        // Take the receiver out before awaiting so the mutex guard never crosses
        // the `.await`.
        let rx = self.gate.lock().unwrap().take().expect("transcribe called more than once");
        let _ = rx.await;
        Ok("gated".to_string())
    }
}

const FMT: AudioFormat = AudioFormat { sample_rate: 16_000, channels: 1 };

#[test]
fn accumulates_windows_and_finalizes_once() {
    block_on(async {
        let engine = Buffered::new(CountingTranscriber { format: FMT });
        engine.begin_utterance(FMT).await.unwrap();

        // Two windows accumulate; neither yields an event.
        assert_eq!(engine.feed(&[0.0; 3]).await.unwrap(), vec![]);
        assert_eq!(engine.feed(&[0.0; 5]).await.unwrap(), vec![]);

        // The whole 8-sample buffer transcribes as one Final.
        let events = engine.end_utterance().await.unwrap();
        assert_eq!(events, vec![SttEvent::Final("heard 8 samples".into())]);
    });
}

#[test]
fn begin_utterance_clears_the_previous_buffer() {
    block_on(async {
        let engine = Buffered::new(CountingTranscriber { format: FMT });

        engine.begin_utterance(FMT).await.unwrap();
        engine.feed(&[0.0; 9]).await.unwrap();
        // A fresh begin drops the 9 accumulated samples.
        engine.begin_utterance(FMT).await.unwrap();
        engine.feed(&[0.0; 2]).await.unwrap();

        let events = engine.end_utterance().await.unwrap();
        assert_eq!(events, vec![SttEvent::Final("heard 2 samples".into())]);
    });
}

#[test]
fn unsupported_format_surfaces_from_end_not_begin() {
    block_on(async {
        let engine = Buffered::new(CountingTranscriber { format: FMT });
        // begin accepts any format — validation is deferred to the one-shot engine.
        let wrong = AudioFormat::new(48_000, 2);
        engine.begin_utterance(wrong).await.expect("begin does not validate the format");
        engine.feed(&[0.0; 4]).await.unwrap();

        match engine.end_utterance().await {
            Err(SttError::UnsupportedFormat { expected, got }) => {
                assert_eq!(expected, FMT);
                assert_eq!(got, wrong);
            }
            other => panic!("expected a deferred format rejection, got {other:?}"),
        }
    });
}

#[test]
fn cancel_discards_the_pending_utterance() {
    block_on(async {
        let engine = Buffered::new(CountingTranscriber { format: FMT });
        engine.begin_utterance(FMT).await.unwrap();
        engine.feed(&[0.0; 4]).await.unwrap();

        engine.cancel();

        // After cancel there is no active utterance: feed and end both reject
        // until a new begin.
        assert!(matches!(engine.feed(&[0.0; 4]).await, Err(SttError::Engine(_))));
        assert!(matches!(engine.end_utterance().await, Err(SttError::Engine(_))));

        // A fresh utterance starts clean — only the new samples count.
        engine.begin_utterance(FMT).await.unwrap();
        engine.feed(&[0.0; 1]).await.unwrap();
        let events = engine.end_utterance().await.unwrap();
        assert_eq!(events, vec![SttEvent::Final("heard 1 samples".into())]);
    });
}

#[test]
fn cancel_is_idempotent() {
    block_on(async {
        let engine = Buffered::new(CountingTranscriber { format: FMT });
        engine.begin_utterance(FMT).await.unwrap();
        engine.feed(&[0.0; 4]).await.unwrap();
        engine.cancel();
        engine.cancel(); // second cancel is a no-op, not a panic
        assert!(matches!(engine.end_utterance().await, Err(SttError::Engine(_))));
    });
}

#[test]
fn cancel_discards_an_in_flight_result() {
    block_on(async {
        let (tx, rx) = oneshot::channel::<()>();
        let engine = Buffered::new(GatedTranscriber { gate: Mutex::new(Some(rx)) });
        engine.begin_utterance(FMT).await.unwrap();
        engine.feed(&[0.0; 4]).await.unwrap();

        // Drive end_utterance (which parks on the gate) concurrently with a
        // canceller. `join!` polls end first — it snapshots the generation and
        // awaits the gate — then runs the second branch, which cancels (bumping
        // the generation) and opens the gate. end then resumes, sees the changed
        // generation, and discards its stale transcript.
        let (events, _) = futures::join!(engine.end_utterance(), async {
            engine.cancel();
            let _ = tx.send(());
        });

        assert_eq!(
            events.unwrap(),
            vec![],
            "a cancel during inference must discard the stale Final"
        );
    });
}
