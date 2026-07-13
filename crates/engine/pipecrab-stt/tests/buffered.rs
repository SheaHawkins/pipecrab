//! `Buffered` adapts a one-shot `Transcriber` to the `StreamingTranscriber`
//! protocol: it accumulates `feed`s (emitting no partials) and produces a single
//! `Final` on `end_utterance`. These tests pin that behavior, its one honest
//! limitation (stale-result discard on cancel), and that it reports the wrapped
//! engine's `input_format`.
//!
//! Deterministic and tokio-free (`block_on`), so it rides the default
//! `cargo test --workspace` path.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::channel::oneshot;
use futures::executor::block_on;
use pipecrab_core::AudioFormat;
use pipecrab_stt::{Buffered, StreamingTranscriber, SttError, SttEvent, Transcriber};

/// A hardware-free one-shot transcriber: reports the sample count it was handed
/// and declares its configured format.
struct CountingTranscriber {
    format: AudioFormat,
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Transcriber for CountingTranscriber {
    fn input_format(&self) -> AudioFormat {
        self.format
    }

    async fn transcribe(&self, samples: Arc<[f32]>) -> Result<String, SttError> {
        Ok(format!("heard {} samples", samples.len()))
    }
}

struct RetainingTranscriber {
    received: Arc<Mutex<Option<Arc<[f32]>>>>,
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Transcriber for RetainingTranscriber {
    fn input_format(&self) -> AudioFormat {
        FMT
    }

    async fn transcribe(&self, samples: Arc<[f32]>) -> Result<String, SttError> {
        *self.received.lock().unwrap() = Some(samples);
        Ok(String::new())
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
    fn input_format(&self) -> AudioFormat {
        FMT
    }

    async fn transcribe(&self, _samples: Arc<[f32]>) -> Result<String, SttError> {
        // Take the receiver out before awaiting so the mutex guard never crosses
        // the `.await`.
        let rx = self
            .gate
            .lock()
            .unwrap()
            .take()
            .expect("transcribe called more than once");
        let _ = rx.await;
        Ok("gated".to_string())
    }
}

const FMT: AudioFormat = AudioFormat {
    sample_rate: 16_000,
    channels: 1,
};

#[test]
fn input_format_delegates_to_the_inner_transcriber() {
    let engine = Buffered::new(CountingTranscriber {
        format: AudioFormat::new(48_000, 2),
    });
    assert_eq!(engine.input_format(), AudioFormat::new(48_000, 2));
}

#[test]
fn a_single_chunk_reaches_the_transcriber_without_copying() {
    block_on(async {
        let received = Arc::new(Mutex::new(None));
        let engine = Buffered::new(RetainingTranscriber {
            received: received.clone(),
        });
        let samples: Arc<[f32]> = Arc::from([0.25, -0.25]);
        let retained = samples.clone();

        engine.begin_utterance().await.unwrap();
        engine.feed(samples).await.unwrap();
        engine.end_utterance().await.unwrap();

        assert!(Arc::ptr_eq(
            &retained,
            received.lock().unwrap().as_ref().unwrap()
        ));
    });
}

#[test]
fn accumulates_windows_and_finalizes_once() {
    block_on(async {
        let engine = Buffered::new(CountingTranscriber { format: FMT });
        engine.begin_utterance().await.unwrap();

        // Two windows accumulate; neither yields an event.
        assert_eq!(engine.feed(Arc::from([0.0; 3])).await.unwrap(), vec![]);
        assert_eq!(engine.feed(Arc::from([0.0; 5])).await.unwrap(), vec![]);

        // The whole 8-sample buffer transcribes as one Final.
        let events = engine.end_utterance().await.unwrap();
        assert_eq!(events, vec![SttEvent::Final("heard 8 samples".into())]);
    });
}

#[test]
fn begin_utterance_rejects_a_second_begin() {
    block_on(async {
        let engine = Buffered::new(CountingTranscriber { format: FMT });

        engine.begin_utterance().await.unwrap();
        engine.feed(Arc::from([0.0; 9])).await.unwrap();

        // A second begin without closing the first is a protocol violation:
        // reject it rather than silently dropping the 9 accumulated samples.
        assert!(matches!(
            engine.begin_utterance().await,
            Err(SttError::Engine(_))
        ));

        // The original utterance is untouched and still finalizes over its 9.
        let events = engine.end_utterance().await.unwrap();
        assert_eq!(events, vec![SttEvent::Final("heard 9 samples".into())]);

        // Once closed, a fresh begin is accepted again.
        engine.begin_utterance().await.unwrap();
    });
}

#[test]
fn feed_and_end_without_begin_are_protocol_errors() {
    block_on(async {
        let engine = Buffered::new(CountingTranscriber { format: FMT });
        // With no active utterance, feed and end both surface the protocol
        // violation rather than silently accepting stray audio — this is how an
        // upstream contract breach (audio ahead of any SpeechStarted) becomes a
        // loud, recoverable engine error at the stage.
        assert!(matches!(
            engine.feed(Arc::from([0.0; 4])).await,
            Err(SttError::Engine(_))
        ));
        assert!(matches!(
            engine.end_utterance().await,
            Err(SttError::Engine(_))
        ));
    });
}

#[test]
fn cancel_discards_the_pending_utterance() {
    block_on(async {
        let engine = Buffered::new(CountingTranscriber { format: FMT });
        engine.begin_utterance().await.unwrap();
        engine.feed(Arc::from([0.0; 4])).await.unwrap();

        engine.cancel();

        // After cancel there is no active utterance: feed and end both reject
        // until a new begin.
        assert!(matches!(
            engine.feed(Arc::from([0.0; 4])).await,
            Err(SttError::Engine(_))
        ));
        assert!(matches!(
            engine.end_utterance().await,
            Err(SttError::Engine(_))
        ));

        // A fresh utterance starts clean — only the new samples count.
        engine.begin_utterance().await.unwrap();
        engine.feed(Arc::from([0.0; 1])).await.unwrap();
        let events = engine.end_utterance().await.unwrap();
        assert_eq!(events, vec![SttEvent::Final("heard 1 samples".into())]);
    });
}

#[test]
fn cancel_is_idempotent() {
    block_on(async {
        let engine = Buffered::new(CountingTranscriber { format: FMT });
        engine.begin_utterance().await.unwrap();
        engine.feed(Arc::from([0.0; 4])).await.unwrap();
        engine.cancel();
        engine.cancel(); // second cancel is a no-op, not a panic
        assert!(matches!(
            engine.end_utterance().await,
            Err(SttError::Engine(_))
        ));
    });
}

#[test]
fn cancel_discards_an_in_flight_result() {
    block_on(async {
        let (tx, rx) = oneshot::channel::<()>();
        let engine = Buffered::new(GatedTranscriber {
            gate: Mutex::new(Some(rx)),
        });
        engine.begin_utterance().await.unwrap();
        engine.feed(Arc::from([0.0; 4])).await.unwrap();

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
