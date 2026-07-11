//! `SttStage` v2 adapts a `StreamingTranscriber` into a stage, gated by the VAD's
//! `SpeechStarted` / `SpeechStopped` edges and fronted by a pre-roll ring that
//! keeps an utterance's onset.
//!
//! The behavior table is defined in terms of the synchronous `decide_*` output,
//! so most tests drive `decide_data` / `decide_system` directly and assert on the
//! emitted effects — fully deterministic, no lane-ordering races. Two async tests
//! cover the parts that only exist past the decide step: the `SttEvent` →
//! `Transcript` mapping in `perform`, and barge-in through the real run loop.
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
    SttConfig, SttEffect, SttError, SttEvent, SttStage, StreamingTranscriber,
};

/// What a [`Mock`] recorded, shared with the test via an `Arc<Mutex<_>>`.
#[derive(Default)]
struct Log {
    /// Formats passed to `begin_utterance`, in order.
    begins: Vec<AudioFormat>,
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

/// A hardware-free [`StreamingTranscriber`]: records every call, emits a partial
/// per feed and a final on end, and optionally parks its first `feed` so a
/// barge-in can drop it in flight.
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
    async fn begin_utterance(&self, format: AudioFormat) -> Result<(), SttError> {
        self.log.lock().unwrap().begins.push(format);
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
            SttEffect::Begin(f) => format!("begin {}/{}", f.sample_rate, f.channels),
            SttEffect::Feed(c) => format!("feed {}", c.samples.len()),
            SttEffect::End => "end".to_string(),
        })
        .collect()
}

/// A config with an explicit pre-roll budget in milliseconds.
fn config_ms(ms: u64) -> SttConfig {
    SttConfig { preroll: std::time::Duration::from_millis(ms) }
}

#[test]
fn preroll_evicts_by_duration_and_feeds_in_arrival_order_before_live() {
    // 1000 Hz mono makes 1 sample == 1 ms, so a 100 ms budget holds ~100 samples.
    let mut stage = SttStage::with_config(Mock::silent(), config_ms(100));

    // Mixed chunk sizes stream in while idle. Running the eviction by hand:
    //   +20        -> [20]        (20 ms)
    //   +50        -> [20,50]     (70 ms)
    //   +40 (110)  -> evict 20    -> [50,40]    (90 ms)
    //   +30 (120)  -> evict 50    -> [40,30]    (70 ms)
    // so only the last two chunks — 40 then 30 — survive, in arrival order.
    for n in [20usize, 50, 40, 30] {
        let d = stage.decide_data(&audio(1000, 1, n));
        assert_eq!(d.disposition, Disposition::Drop, "idle audio is consumed into the ring");
        assert!(d.effects.is_empty(), "idle audio emits nothing until speech starts");
    }

    // SpeechStarted drains the ring: begin, then the survivors in arrival order.
    let started = stage.decide_system(Direction::Down, &SystemFrame::SpeechStarted);
    assert_eq!(started.disposition, Disposition::Forward, "the edge is forwarded downstream");
    assert_eq!(summarize(&started.effects), vec!["begin 1000/1", "feed 40", "feed 30"]);

    // A live chunk now feeds straight through — after the pre-roll, not before.
    let live = stage.decide_data(&audio(1000, 1, 15));
    assert_eq!(live.disposition, Disposition::Drop);
    assert_eq!(summarize(&live.effects), vec!["feed 15"], "no second begin; pre-roll already fed");
}

#[test]
fn cold_start_with_empty_ring_defers_begin_to_the_first_chunk() {
    let mut stage = SttStage::new(Mock::silent());

    // SpeechStarted with nothing buffered: no format to open with yet, so no
    // effects — but the edge still forwards and the stage is now in speech.
    let started = stage.decide_system(Direction::Down, &SystemFrame::SpeechStarted);
    assert_eq!(started.disposition, Disposition::Forward);
    assert!(started.effects.is_empty(), "an empty ring cold-starts with no begin");

    // The first live chunk opens the utterance from its own format, then feeds.
    let first = stage.decide_data(&audio(16_000, 1, 8));
    assert_eq!(summarize(&first.effects), vec!["begin 16000/1", "feed 8"]);

    // Subsequent chunks feed without re-opening.
    let second = stage.decide_data(&audio(16_000, 1, 5));
    assert_eq!(summarize(&second.effects), vec!["feed 5"]);

    // SpeechStopped closes the (now-open) utterance.
    let stopped = stage.decide_system(Direction::Down, &SystemFrame::SpeechStopped);
    assert_eq!(stopped.disposition, Disposition::Forward);
    assert_eq!(summarize(&stopped.effects), vec!["end"]);
}

#[test]
fn duplicate_speech_edges_are_idempotent_both_ways() {
    let mut stage = SttStage::new(Mock::silent());
    stage.decide_data(&audio(16_000, 1, 4)); // one chunk of pre-roll

    // First SpeechStarted opens the utterance...
    let first = stage.decide_system(Direction::Down, &SystemFrame::SpeechStarted);
    assert_eq!(summarize(&first.effects), vec!["begin 16000/1", "feed 4"]);
    // ...a duplicate one is a no-op: forwarded, but no second begin_utterance.
    let dup_start = stage.decide_system(Direction::Down, &SystemFrame::SpeechStarted);
    assert_eq!(dup_start.disposition, Disposition::Forward);
    assert!(dup_start.effects.is_empty(), "a repeated start must not begin a second utterance");

    // First SpeechStopped closes it...
    let stop = stage.decide_system(Direction::Down, &SystemFrame::SpeechStopped);
    assert_eq!(summarize(&stop.effects), vec!["end"]);
    // ...a duplicate one is a no-op: forwarded, no second end_utterance.
    let dup_stop = stage.decide_system(Direction::Down, &SystemFrame::SpeechStopped);
    assert_eq!(dup_stop.disposition, Disposition::Forward);
    assert!(dup_stop.effects.is_empty(), "a repeated stop must not end a second time");
}

#[test]
fn format_change_while_idle_clears_the_ring() {
    let mut stage = SttStage::new(Mock::silent());

    // Two chunks accumulate in 16 kHz mono...
    stage.decide_data(&audio(16_000, 1, 10));
    stage.decide_data(&audio(16_000, 1, 10));
    // ...then a chunk in a different format arrives. The ring assumes one uniform
    // format, so it drops the stale contents and restarts in the new one.
    stage.decide_data(&audio(48_000, 2, 8));

    let started = stage.decide_system(Direction::Down, &SystemFrame::SpeechStarted);
    assert_eq!(
        summarize(&started.effects),
        vec!["begin 48000/2", "feed 8"],
        "only the post-change chunk survives, and begin uses its format",
    );
}

#[test]
fn interrupt_cancels_resets_and_keeps_the_ring() {
    let mock = Mock::silent();
    let log = mock.log.clone();
    let mut stage = SttStage::new(mock);

    // A chunk of pre-roll accumulates while idle.
    stage.decide_data(&audio(16_000, 1, 12));

    // An interrupt fires the cancel control-call, forwards, and emits nothing.
    let interrupt = stage.decide_system(Direction::Down, &SystemFrame::Interrupt);
    assert_eq!(interrupt.disposition, Disposition::Forward);
    assert!(interrupt.effects.is_empty(), "interrupt cancels via a control-call, not an effect");
    assert_eq!(log.lock().unwrap().cancels, 1, "the engine's cancel flag was flipped");

    // The ring survived the interrupt (it only accumulates while idle), so the
    // next SpeechStarted still replays that pre-roll.
    let started = stage.decide_system(Direction::Down, &SystemFrame::SpeechStarted);
    assert_eq!(summarize(&started.effects), vec!["begin 16000/1", "feed 12"]);

    // A mid-speech interrupt resets the in-speech state cleanly: audio that
    // follows is buffered as fresh pre-roll again, not fed to a torn utterance.
    stage.decide_system(Direction::Down, &SystemFrame::Interrupt);
    assert_eq!(log.lock().unwrap().cancels, 2);
    let idle_again = stage.decide_data(&audio(16_000, 1, 7));
    assert_eq!(idle_again.disposition, Disposition::Drop);
    assert!(idle_again.effects.is_empty(), "after interrupt the stage is idle: audio goes to the ring");
    let restarted = stage.decide_system(Direction::Down, &SystemFrame::SpeechStarted);
    assert_eq!(summarize(&restarted.effects), vec!["begin 16000/1", "feed 7"], "clean restart");
}

#[test]
fn events_map_to_user_transcripts() {
    block_on(async {
        let stage = SttStage::new(Mock::silent());
        let (out, mut inbound) = link(8);
        let fmt = AudioFormat::new(16_000, 1);

        // Drive the three effects perform interprets, in utterance order.
        stage.perform(SttEffect::Begin(fmt), &out).await.unwrap();
        let chunk = AudioChunk::new(Arc::from(&[0.0f32; 4][..]), fmt);
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
fn barge_in_drops_the_in_flight_feed_and_cancels() {
    // The run loop's job here is what no decide-level test can show: an Interrupt
    // arriving mid-`feed` drops that in-flight `perform` and the stage cancels the
    // engine from `decide_system`. (That the *next* utterance then runs clean —
    // no torn state — is pinned deterministically by
    // `interrupt_cancels_resets_and_keeps_the_ring`; replaying a second utterance
    // through the pipeline here would race the interrupt's data-lane flush.)
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
            // Cold start, then one live chunk whose feed opens the utterance and
            // then parks.
            let _ = input.send_system(Direction::Down, SystemFrame::SpeechStarted).await;
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
        assert_eq!(log.begins.len(), 1, "the utterance opened before the feed parked");
        assert!(log.feeds.is_empty(), "the feed was dropped before it recorded anything");
        assert_eq!(log.ends, 0, "a cancelled utterance is never ended");
        assert_eq!(finals, 0, "the dropped feed produced no transcript");
    });
}
