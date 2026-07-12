//! `SttStage` is a **stateless** adapter from a `StreamingTranscriber` to a
//! stage, driven by the VAD gate's `SpeechStarted` / `SpeechStopped` edges. Under
//! the gate contract the edges bracket the utterance audio — `SpeechStarted`
//! precedes every chunk, `SpeechStopped` follows the last — so the stage just
//! translates: edge → begin/end, chunk → feed.
//!
//! The edges ride the data lane, so they are handled in `decide_data` alongside
//! audio; only `Interrupt` remains a `decide_system` frame. The behavior table is
//! defined in terms of that synchronous `decide_*` output, so the deterministic
//! tests drive `decide_data` / `decide_system` directly and assert on the emitted
//! effects — no lane-ordering races. The remaining tests use the real run loop
//! for the parts that only exist past the decide step: the `SttEvent` →
//! `Transcript` mapping in `perform`, the fatal format teardown, a surfaced
//! protocol violation, and barge-in.
//!
//! Deterministic and tokio-free (`block_on`), so it rides the default
//! `cargo test --workspace` path.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::executor::block_on;
use futures::stream::StreamExt;
use pipecrab_core::{
    AudioChunk, AudioFormat, DataFrame, Direction, Disposition, Finality, Processor, Role,
    SystemFrame,
};
use pipecrab_runtime::{link, PipelineBuilder, Received, Stage};
use pipecrab_stt::{
    Buffered, SttEffect, SttError, SttEvent, SttStage, StreamingTranscriber, Transcriber,
};

/// The format the mocks declare; the stage caches it and enforces it.
const FMT: AudioFormat = AudioFormat { sample_rate: 16_000, channels: 1 };

/// What a [`Mock`] recorded, shared with the test via an `Arc<Mutex<_>>`.
#[derive(Default)]
struct Log {
    /// Number of `begin_utterance` calls.
    begins: usize,
    /// Sample count of each `feed`, in order.
    feeds: Vec<usize>,
    /// Number of `end_utterance` calls.
    ends: usize,
    /// Number of `cancel` control-calls.
    cancels: usize,
}

/// A milestone the [`Mock`] reports to a pipeline test so it can advance the
/// input script only once the engine has reached a known point — the antidote to
/// the sys-before-data lane priority reordering the frames.
#[derive(Debug, PartialEq, Eq)]
enum Note {
    /// The gated first `feed` has started and is now parked.
    FeedStarted,
    /// A `feed` recorded `n` samples.
    Fed(usize),
}

/// A hardware-free [`StreamingTranscriber`]: declares [`FMT`], records every
/// call, emits a partial per feed and a final on end, and optionally parks its
/// first `feed` so a barge-in can drop it in flight.
struct Mock {
    log: Arc<Mutex<Log>>,
    notes: mpsc::UnboundedSender<Note>,
    /// If present, the first `feed` awaits this (never fired) and is unparked
    /// only by the interrupt dropping its future.
    park: Mutex<Option<oneshot::Receiver<()>>>,
}

impl Mock {
    /// A mock whose notes go nowhere — for the synchronous decide/perform tests.
    fn silent() -> Self {
        let (notes, _rx) = mpsc::unbounded();
        Self { log: Arc::new(Mutex::new(Log::default())), notes, park: Mutex::new(None) }
    }

    /// A mock that reports milestones on `notes`; `park` gates its first feed.
    fn reporting(notes: mpsc::UnboundedSender<Note>, park: Option<oneshot::Receiver<()>>) -> Self {
        Self { log: Arc::new(Mutex::new(Log::default())), notes, park: Mutex::new(park) }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl StreamingTranscriber for Mock {
    fn input_format(&self) -> AudioFormat {
        FMT
    }

    async fn begin_utterance(&self) -> Result<(), SttError> {
        self.log.lock().unwrap().begins += 1;
        Ok(())
    }

    async fn feed(&self, samples: &[f32]) -> Result<Vec<SttEvent>, SttError> {
        // The gated first feed announces itself and parks until dropped: the
        // barge-in test relies on the interrupt dropping this future right here.
        let park = self.park.lock().unwrap().take();
        if let Some(rx) = park {
            let _ = self.notes.unbounded_send(Note::FeedStarted);
            let _ = rx.await;
        }
        let n = samples.len();
        let total = {
            let mut log = self.log.lock().unwrap();
            log.feeds.push(n);
            log.feeds.iter().sum::<usize>()
        };
        let _ = self.notes.unbounded_send(Note::Fed(n));
        Ok(vec![SttEvent::Partial { text: format!("partial {total}").into(), stable: 0 }])
    }

    async fn end_utterance(&self) -> Result<Vec<SttEvent>, SttError> {
        let total = {
            let mut log = self.log.lock().unwrap();
            log.ends += 1;
            log.feeds.iter().sum::<usize>()
        };
        Ok(vec![SttEvent::Final(format!("heard {total} samples").into())])
    }

    fn cancel(&self) {
        self.log.lock().unwrap().cancels += 1;
    }
}

/// A hardware-free one-shot [`Transcriber`]: declares [`FMT`] and echoes its
/// sample count. Wrapped in [`Buffered`] to exercise the stage against a *real*
/// adapter that surfaces protocol violations.
struct OneShot;

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Transcriber for OneShot {
    fn input_format(&self) -> AudioFormat {
        FMT
    }

    async fn transcribe(&self, samples: &[f32]) -> Result<String, SttError> {
        Ok(format!("heard {} samples", samples.len()))
    }
}

/// An `Audio` data frame of `n` zeroed interleaved samples in the given format.
fn audio(sample_rate: u32, channels: u16, n: usize) -> DataFrame {
    let chunk = AudioChunk::new(Arc::from(vec![0.0f32; n]), AudioFormat::new(sample_rate, channels));
    DataFrame::Audio(chunk)
}

/// Render a stage's emitted effects as compact strings for readable assertions.
fn summarize(effects: &[SttEffect]) -> Vec<String> {
    effects
        .iter()
        .map(|e| match e {
            SttEffect::Begin => "begin".to_string(),
            SttEffect::Feed(c) => format!("feed {}", c.samples.len()),
            SttEffect::End => "end".to_string(),
            SttEffect::RejectFormat { got } => {
                format!("reject {}/{}", got.sample_rate, got.channels)
            }
        })
        .collect()
}

#[test]
fn happy_path_translates_edges_and_chunks_to_the_utterance_protocol() {
    // The gate contract: SpeechStarted, then the utterance's chunks, then
    // SpeechStopped. The stage translates that one-for-one: begin, a feed per
    // chunk in arrival order, end. No state, no pre-roll — the gate already
    // bracketed the onset audio in behind the edge.
    let mut stage = SttStage::new(Mock::silent());

    let started = stage.decide_data(&DataFrame::SpeechStarted);
    assert_eq!(started.disposition, Disposition::Forward, "the edge is forwarded downstream");
    assert_eq!(summarize(&started.effects), vec!["begin"]);

    // Three conforming chunks each feed straight through, unconditionally.
    for n in [10usize, 20, 30] {
        let live = stage.decide_data(&audio(16_000, 1, n));
        assert_eq!(live.disposition, Disposition::Drop, "consumed audio does not travel on");
        assert_eq!(summarize(&live.effects), vec![format!("feed {n}")]);
    }

    let stopped = stage.decide_data(&DataFrame::SpeechStopped);
    assert_eq!(stopped.disposition, Disposition::Forward, "the edge is forwarded downstream");
    assert_eq!(summarize(&stopped.effects), vec!["end"]);
}

#[test]
fn stateless_stage_does_not_dedup_a_duplicate_begin() {
    // This pins statelessness — it is NOT a blessing of double-begins. A second
    // SpeechStarted violates the gate's alternation contract; the point here is
    // only that the stage keeps no `in_speech` latch, so it does not *absorb*
    // (dedup) the duplicate — it emits a second `begin` and leaves the violation
    // for the engine to reject. That rejection is proven end-to-end in
    // `duplicate_speech_started_surfaces_a_recoverable_engine_error`.
    let mut stage = SttStage::new(Mock::silent());
    let first = stage.decide_data(&DataFrame::SpeechStarted);
    let second = stage.decide_data(&DataFrame::SpeechStarted);
    assert_eq!(summarize(&first.effects), vec!["begin"]);
    assert_eq!(
        summarize(&second.effects),
        vec!["begin"],
        "no latch: the effect is emitted again, not deduped",
    );
}

#[test]
fn nonconforming_chunk_cancels_and_emits_reject() {
    // The synchronous half of the fatal path: a mismatch cancels the engine
    // (hygiene) and emits RejectFormat. The fatal teardown itself is a perform
    // concern, pinned by the run-loop tests below.
    let mock = Mock::silent();
    let log = mock.log.clone();
    let mut stage = SttStage::new(mock);

    let rejected = stage.decide_data(&audio(48_000, 2, 8));
    assert_eq!(rejected.disposition, Disposition::Drop, "the nonconforming chunk is consumed");
    assert_eq!(summarize(&rejected.effects), vec!["reject 48000/2"]);
    assert_eq!(log.lock().unwrap().cancels, 1, "the engine is cancelled before the reject");
}

#[test]
fn interrupt_cancels_unconditionally() {
    let mock = Mock::silent();
    let log = mock.log.clone();
    let mut stage = SttStage::new(mock);

    // While idle: an interrupt still fires the cancel control-call, forwards, and
    // emits nothing. It is idempotent, so cancelling with no open utterance is a
    // harmless no-op.
    let idle = stage.decide_system(Direction::Down, &SystemFrame::Interrupt);
    assert_eq!(idle.disposition, Disposition::Forward);
    assert!(idle.effects.is_empty(), "interrupt cancels via a control-call, not an effect");
    assert_eq!(log.lock().unwrap().cancels, 1);

    // Mid-utterance (after a begin): the same unconditional cancel.
    stage.decide_data(&DataFrame::SpeechStarted);
    let mid = stage.decide_system(Direction::Down, &SystemFrame::Interrupt);
    assert_eq!(mid.disposition, Disposition::Forward);
    assert!(mid.effects.is_empty());
    assert_eq!(log.lock().unwrap().cancels, 2, "cancel is unconditional, idle or mid-utterance");
}

#[test]
fn events_map_to_user_transcripts() {
    block_on(async {
        let stage = SttStage::new(Mock::silent());
        let (out, mut inbound) = link(8);

        // Drive the three effects perform interprets, in utterance order.
        stage.perform(SttEffect::Begin, &out).await.unwrap();
        let chunk = AudioChunk::new(Arc::from(&[0.0f32; 4][..]), FMT);
        stage.perform(SttEffect::Feed(chunk), &out).await.unwrap();
        stage.perform(SttEffect::End, &out).await.unwrap();

        // Dropping `out` closes the lane so the drain terminates.
        drop(out);
        let mut transcripts = Vec::new();
        while let Some(frame) = inbound.data.next().await {
            if let DataFrame::Transcript(t) = frame {
                transcripts.push(t);
            }
        }

        // feed's Partial -> user_partial; end's Final -> user_final. Begin emits
        // nothing downstream.
        assert_eq!(transcripts.len(), 2, "one partial from feed, one final from end");
        let partial = &transcripts[0];
        assert_eq!(partial.role, Role::User);
        assert_eq!(partial.finality, Finality::Partial { stable: 0 });
        assert_eq!(&*partial.text, "partial 4");
        let final_t = &transcripts[1];
        assert_eq!(final_t.role, Role::User);
        assert_eq!(final_t.finality, Finality::Final);
        assert_eq!(&*final_t.text, "heard 4 samples");
    });
}

#[test]
fn first_nonconforming_chunk_completes_with_a_fatal_error() {
    block_on(async {
        // The engine declares 16 kHz mono; the very first chunk is 48 kHz stereo.
        let mock = Mock::silent();
        let log = mock.log.clone();
        let (ends, driver) = PipelineBuilder::new().stage(SttStage::new(mock)).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let _ = input.send_data(audio(48_000, 2, 4)).await;
        };

        let drain = async move {
            let mut error = None;
            while let Some(received) = output.recv().await {
                if let Received::Sys(Direction::Up, SystemFrame::Error { message, fatal }) = received
                {
                    error = Some((message, fatal));
                }
            }
            error
        };

        let (_, error, _) = futures::join!(feed, drain, driver);
        let (message, fatal) = error.expect("a format mismatch should surface an Error frame");
        assert!(fatal, "a format mismatch is fatal: the stage can never conform the audio");
        assert!(message.contains("SttStage requires"), "unexpected message: {message}");
        assert_eq!(log.lock().unwrap().cancels, 1, "the engine was cancelled before the reject");
    });
}

#[test]
fn mismatch_mid_utterance_cancels_and_tears_down_fatally() {
    block_on(async {
        // Open a real utterance, then feed a nonconforming chunk: the mismatch is
        // still fatal, and the engine is cancelled first so no worker is left
        // mid-utterance.
        let mock = Mock::silent();
        let log = mock.log.clone();
        let (ends, driver) = PipelineBuilder::new().stage(SttStage::new(mock)).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let _ = input.send_data(DataFrame::SpeechStarted).await;
            let _ = input.send_data(audio(16_000, 1, 8)).await; // conforming
            let _ = input.send_data(audio(48_000, 2, 8)).await; // mismatch
        };

        let drain = async move {
            let mut error = None;
            while let Some(received) = output.recv().await {
                if let Received::Sys(Direction::Up, SystemFrame::Error { message, fatal }) = received
                {
                    error = Some((message, fatal));
                }
            }
            error
        };

        let (_, error, _) = futures::join!(feed, drain, driver);
        let (message, fatal) = error.expect("a mid-utterance mismatch should surface an Error frame");
        assert!(fatal, "a format mismatch is fatal wherever it arrives");
        assert!(message.contains("SttStage requires"), "unexpected message: {message}");
        let log = log.lock().unwrap();
        assert_eq!(log.begins, 1, "the utterance opened before the mismatch");
        assert_eq!(log.cancels, 1, "the mismatch cancelled the open utterance");
    });
}

#[test]
fn audio_before_speech_started_surfaces_a_recoverable_engine_error() {
    block_on(async {
        // The stage trusts the gate contract rather than policing it. If a
        // malformed upstream sends audio ahead of any SpeechStarted, the stage
        // feeds it and the engine — here a real `Buffered`, with no open
        // utterance — surfaces a feed-without-begin protocol error. It is loud
        // and recoverable, not silently absorbed.
        let stage = SttStage::new(Buffered::new(OneShot));
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let _ = input.send_data(audio(16_000, 1, 4)).await;
        };

        let drain = async move {
            let mut error = None;
            while let Some(received) = output.recv().await {
                if let Received::Sys(Direction::Up, SystemFrame::Error { message, fatal }) = received
                {
                    error = Some((message, fatal));
                }
            }
            error
        };

        let (_, error, _) = futures::join!(feed, drain, driver);
        let (message, fatal) = error.expect("the protocol violation should surface an Error frame");
        assert!(!fatal, "an engine protocol error is recoverable, not fatal");
        assert!(
            message.contains("without an active utterance"),
            "unexpected message: {message}",
        );
    });
}

#[test]
fn duplicate_speech_started_surfaces_a_recoverable_engine_error() {
    block_on(async {
        // The mirror of the feed-before-begin case, and the end-to-end proof that
        // the "protocol trust, not defense" stance does not swallow a double
        // begin. The stage keeps no latch, so a duplicate SpeechStarted flows
        // through as a second `begin`; a real `Buffered` already has an utterance
        // open, so it rejects — loud and recoverable, not silently absorbed (a
        // resetting engine would otherwise discard the first utterance's audio).
        let stage = SttStage::new(Buffered::new(OneShot));
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let _ = input.send_data(DataFrame::SpeechStarted).await;
            let _ = input.send_data(DataFrame::SpeechStarted).await; // contract violation
        };

        let drain = async move {
            let mut error = None;
            while let Some(received) = output.recv().await {
                if let Received::Sys(Direction::Up, SystemFrame::Error { message, fatal }) = received
                {
                    error = Some((message, fatal));
                }
            }
            error
        };

        let (_, error, _) = futures::join!(feed, drain, driver);
        let (message, fatal) = error.expect("a duplicate begin should surface an Error frame");
        assert!(!fatal, "a protocol violation is recoverable, not fatal");
        assert!(message.contains("already active"), "unexpected message: {message}");
    });
}

#[test]
fn barge_in_drops_the_in_flight_feed_and_cancels() {
    // The run loop's job here is what no decide-level test can show: an Interrupt
    // arriving mid-`feed` drops that in-flight `perform` and the stage cancels the
    // engine from `decide_system`.
    block_on(async {
        // The first feed parks on `park_rx`; keeping `park_tx` alive means it only
        // unparks when the interrupt drops its future.
        let (park_tx, park_rx) = oneshot::channel::<()>();
        let (notes_tx, mut notes_rx) = mpsc::unbounded::<Note>();
        let mock = Mock::reporting(notes_tx, Some(park_rx));
        let log = mock.log.clone();
        let stage = SttStage::new(mock);

        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _park_tx = park_tx; // keep the feed parked until the interrupt drops it
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            // Under the gate contract the edge leads: SpeechStarted opens the
            // utterance (`begin`), then the chunk feeds — and that first feed
            // parks.
            let _ = input.send_data(DataFrame::SpeechStarted).await;
            let _ = input.send_data(audio(16_000, 1, 4)).await;
            // Wait until that feed is actually in flight before barging in, so the
            // interrupt lands on a parked perform rather than racing ahead of it.
            assert_eq!(notes_rx.next().await, Some(Note::FeedStarted));
            let _ = input.send_system(Direction::Down, SystemFrame::Interrupt).await;
            // Returning drops `input`, cascading shutdown through the pipeline.
        };

        let drain = async move {
            let mut finals = 0;
            while let Some(received) = output.recv().await {
                if let Received::Data(DataFrame::Transcript(t)) = received {
                    if t.finality == Finality::Final {
                        finals += 1;
                    }
                }
            }
            finals
        };

        let (_, finals, _) = futures::join!(feed, drain, driver);

        let log = log.lock().unwrap();
        assert_eq!(log.cancels, 1, "the barge-in flipped the engine's cancel flag once");
        assert_eq!(log.begins, 1, "the utterance opened before the feed parked");
        assert!(log.feeds.is_empty(), "the feed was dropped before it recorded anything");
        assert_eq!(log.ends, 0, "a cancelled utterance is never ended");
        assert_eq!(finals, 0, "the dropped feed produced no transcript");
    });
}
