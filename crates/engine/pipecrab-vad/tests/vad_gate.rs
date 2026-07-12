//! Tests for the [`VadStage`](pipecrab_vad::VadStage) gate.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::executor::block_on;
use futures::stream::StreamExt;
use pipecrab_core::{AudioChunk, AudioFormat, DataFrame, Direction, Processor, SystemFrame};
use pipecrab_runtime::{link, PipelineBuilder, Received, Stage};
use pipecrab_vad::{GateConfig, VadError, VadEvent, VadStage, VoiceActivityDetector};

/// Replays edge batches and records reset calls.
struct MockDetector {
    script: Mutex<VecDeque<Vec<VadEvent>>>,
    format: AudioFormat,
    resets: Arc<Mutex<u32>>,
}

impl MockDetector {
    fn new(script: Vec<Vec<VadEvent>>, format: AudioFormat) -> Self {
        Self {
            script: Mutex::new(script.into_iter().collect()),
            format,
            resets: Arc::default(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl VoiceActivityDetector for MockDetector {
    fn input_format(&self) -> AudioFormat {
        self.format
    }

    async fn process(&self, _samples: &[f32]) -> Result<Vec<VadEvent>, VadError> {
        Ok(self.script.lock().unwrap().pop_front().unwrap_or_default())
    }

    fn reset(&self) {
        *self.resets.lock().unwrap() += 1;
    }
}

/// A tag for each frame the drain observes, for compact ordering assertions.
#[derive(Debug, PartialEq, Eq)]
enum Seen {
    Started,
    Stopped,
    /// An `Audio` chunk, tagged by its sample count so pre-roll vs trigger vs
    /// live chunks are distinguishable.
    Audio(usize),
}

const FMT: AudioFormat = AudioFormat {
    sample_rate: 16_000,
    channels: 1,
};

fn audio(n: usize) -> DataFrame {
    DataFrame::Audio(AudioChunk::new(Arc::from(vec![0.0f32; n]), FMT))
}

/// Runs edge batches against audio chunks and returns emitted frames and resets.
fn run_gate(
    config: GateConfig,
    script: Vec<Vec<VadEvent>>,
    chunk_sizes: Vec<usize>,
) -> (Vec<Seen>, u32) {
    block_on(async {
        let detector = MockDetector::new(script, FMT);
        let resets = detector.resets.clone();
        let (ends, driver) = PipelineBuilder::new()
            .stage(VadStage::with_config(detector, config))
            .build()
            .start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            for n in chunk_sizes {
                let _ = input.send_data(audio(n)).await;
            }
            // Dropping `input` cascades shutdown through the pipeline.
        };

        let drain = async move {
            let mut seen = Vec::new();
            while let Some(received) = output.recv().await {
                match received {
                    Received::Data(DataFrame::SpeechStarted) => seen.push(Seen::Started),
                    Received::Data(DataFrame::SpeechStopped) => seen.push(Seen::Stopped),
                    Received::Data(DataFrame::Audio(c)) => seen.push(Seen::Audio(c.samples.len())),
                    _ => {}
                }
            }
            seen
        };

        let (_, seen, _) = futures::join!(feed, drain, driver);
        let reset_count = *resets.lock().unwrap();
        (seen, reset_count)
    })
}

fn big_preroll() -> GateConfig {
    // A budget large enough that eviction never fires in these ordering tests.
    GateConfig {
        preroll: std::time::Duration::from_secs(10),
    }
}

#[test]
fn idle_accumulates_and_emits_nothing() {
    // Three silent chunks, no edges: the gate stashes them in the ring and emits
    // nothing at all.
    let (seen, resets) = run_gate(big_preroll(), vec![vec![], vec![], vec![]], vec![1, 2, 3]);
    assert!(seen.is_empty(), "idle silence emits nothing, got {seen:?}");
    assert_eq!(resets, 0);
}

#[test]
fn onset_emits_edge_then_ring_in_arrival_order_then_trigger() {
    // Two idle chunks (10, 20) fill the ring; the third chunk (30) triggers
    // SpeechStarted. Order out: the edge, the ring's chunks in arrival order,
    // then the triggering chunk.
    let script = vec![vec![], vec![], vec![VadEvent::SpeechStarted]];
    let (seen, _) = run_gate(big_preroll(), script, vec![10, 20, 30]);
    assert_eq!(
        seen,
        vec![
            Seen::Started,
            Seen::Audio(10),
            Seen::Audio(20),
            Seen::Audio(30)
        ],
        "onset: edge -> ring (arrival order) -> trigger chunk",
    );
}

#[test]
fn live_speech_passes_through() {
    // Onset on the first chunk, then two live chunks flow straight through as
    // Audio with no further edges.
    let script = vec![vec![VadEvent::SpeechStarted], vec![], vec![]];
    let (seen, _) = run_gate(big_preroll(), script, vec![5, 6, 7]);
    assert_eq!(
        seen,
        vec![
            Seen::Started,
            Seen::Audio(5),
            Seen::Audio(6),
            Seen::Audio(7)
        ],
        "live speech: the trigger chunk then each live chunk pass through",
    );
}

#[test]
fn close_emits_tail_chunk_then_edge() {
    // Onset, a live chunk, then a chunk that closes: the closing chunk is the
    // utterance's tail and must precede the SpeechStopped edge.
    let script = vec![
        vec![VadEvent::SpeechStarted],
        vec![],
        vec![VadEvent::SpeechStopped],
    ];
    let (seen, _) = run_gate(big_preroll(), script, vec![5, 6, 7]);
    assert_eq!(
        seen,
        vec![
            Seen::Started,
            Seen::Audio(5),
            Seen::Audio(6),
            Seen::Audio(7),
            Seen::Stopped
        ],
        "close: tail chunk -> SpeechStopped",
    );
}

#[test]
fn both_edges_in_one_chunk_compose() {
    // A single chunk carries both a start and a stop (a very short blip). The
    // loop composes: edge, ring drain (empty here), trigger chunk once, then the
    // stop edge — the chunk is sent exactly once.
    let script = vec![vec![VadEvent::SpeechStarted, VadEvent::SpeechStopped]];
    let (seen, _) = run_gate(big_preroll(), script, vec![9]);
    assert_eq!(
        seen,
        vec![Seen::Started, Seen::Audio(9), Seen::Stopped],
        "both edges in one chunk: started -> chunk (once) -> stopped",
    );
}

#[test]
fn preroll_evicts_keeping_most_recent_before_onset() {
    // 1 kHz mono makes 1 sample == 1 ms; a 100 ms budget holds ~100 samples.
    // Feed 20, 50, 40, 30 while idle, then trigger on the fifth chunk. Eviction
    // leaves the ring holding 40 then 30 (see the unit test); onset replays those
    // in arrival order ahead of the trigger.
    let config = GateConfig {
        preroll: std::time::Duration::from_millis(100),
    };
    let script = vec![
        vec![],
        vec![],
        vec![],
        vec![],
        vec![VadEvent::SpeechStarted],
    ];

    // These chunks are 1 kHz mono, unlike the 16 kHz `audio()` helper, so drive a
    // bespoke pipeline here.
    let seen = block_on(async {
        let one_khz = AudioFormat::new(1_000, 1);
        let detector = MockDetector::new(script, one_khz);
        let (ends, driver) = PipelineBuilder::new()
            .stage(VadStage::with_config(detector, config))
            .build()
            .start();
        let input = ends.input;
        let mut output = ends.output;
        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            for n in [20usize, 50, 40, 30, 15] {
                let chunk = AudioChunk::new(Arc::from(vec![0.0f32; n]), one_khz);
                let _ = input.send_data(DataFrame::Audio(chunk)).await;
            }
        };
        let drain = async move {
            let mut seen = Vec::new();
            while let Some(received) = output.recv().await {
                match received {
                    Received::Data(DataFrame::SpeechStarted) => seen.push(Seen::Started),
                    Received::Data(DataFrame::Audio(c)) => seen.push(Seen::Audio(c.samples.len())),
                    _ => {}
                }
            }
            seen
        };
        let (_, seen, _) = futures::join!(feed, drain, driver);
        seen
    });
    assert_eq!(
        seen,
        vec![
            Seen::Started,
            Seen::Audio(40),
            Seen::Audio(30),
            Seen::Audio(15)
        ],
        "evicted ring (40,30) replayed in arrival order, then the trigger chunk (15)",
    );
}

// The alternation invariant (events alternate, starting with SpeechStarted) is
// enforced with a `debug_assert!` in the gate, so a misbehaving detector panics
// only in debug builds; gate the test so `cargo test --release` (asserts
// compiled out) does not expect a panic that never fires.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "alternation violated")]
fn back_to_back_started_from_a_misbehaving_detector_panics_loudly() {
    block_on(async {
        // Two SpeechStarted in one batch violates the alternation contract.
        let script = vec![vec![VadEvent::SpeechStarted, VadEvent::SpeechStarted]];
        let detector = MockDetector::new(script, FMT);
        let mut stage = VadStage::new(detector);
        let (out, _inbound) = link(8);

        let decision = stage.decide_data(&audio(4));
        for effect in decision.effects {
            stage.perform(effect, &out).await.unwrap();
        }
    });
}

#[test]
fn format_mismatch_completes_with_a_fatal_error() {
    block_on(async {
        // Detector accepts 16 kHz mono; feed 48 kHz stereo.
        let detector = MockDetector::new(vec![], AudioFormat::new(16_000, 1));
        let (ends, driver) = PipelineBuilder::new()
            .stage(VadStage::new(detector))
            .build()
            .start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let chunk = AudioChunk::new(Arc::from(&[0.0f32; 4][..]), AudioFormat::new(48_000, 2));
            let _ = input.send_data(DataFrame::Audio(chunk)).await;
        };

        let drain = async move {
            let mut error = None;
            while let Some(received) = output.recv().await {
                if let Received::Sys(Direction::Up, SystemFrame::Error { message, fatal }) =
                    received
                {
                    error = Some((message, fatal));
                }
            }
            error
        };

        let (_, error, _) = futures::join!(feed, drain, driver);
        let (message, fatal) = error.expect("a format mismatch should surface an Error frame");
        assert!(
            fatal,
            "a format mismatch is fatal: the gate can never conform the audio"
        );
        assert!(
            message.contains("VadStage requires"),
            "unexpected message: {message}"
        );
    });
}

#[test]
fn interrupt_resets_the_gate_and_the_detector() {
    // An Interrupt clears the ring and calls the detector's `reset` control-call.
    // Because a system Interrupt has lane priority over queued data, this is
    // driven through `decide_*`/`perform` directly (as the behavior table
    // defines it) rather than raced through the run loop.
    block_on(async {
        // process() is called once per audio chunk (the interrupt calls none):
        // chunks 11, 12, 13 are idle ([]), chunk 14 triggers the onset.
        let script = vec![vec![], vec![], vec![], vec![VadEvent::SpeechStarted]];
        let detector = MockDetector::new(script, FMT);
        let resets = detector.resets.clone();
        let mut stage = VadStage::new(detector);
        let (out, mut inbound) = link(16);

        // Helper: run a frame's decided effects through perform.
        async fn drive_data(
            stage: &mut VadStage<MockDetector>,
            frame: DataFrame,
            out: &pipecrab_runtime::Outbound,
        ) {
            let decision = stage.decide_data(&frame);
            for effect in decision.effects {
                stage.perform(effect, out).await.unwrap();
            }
        }

        // Two idle chunks accumulate in the ring.
        drive_data(&mut stage, audio(11), &out).await;
        drive_data(&mut stage, audio(12), &out).await;

        // Interrupt: forwards, emits no effects, clears the ring, resets the detector.
        let interrupt = stage.decide_system(Direction::Down, &SystemFrame::Interrupt);
        assert!(
            interrupt.effects.is_empty(),
            "interrupt resets via control-call, not an effect"
        );
        assert_eq!(
            *resets.lock().unwrap(),
            1,
            "the interrupt fired the detector's reset control-call"
        );

        // Post-reset: one fresh idle chunk, then the onset chunk.
        drive_data(&mut stage, audio(13), &out).await;
        drive_data(&mut stage, audio(14), &out).await;

        drop(out);
        let mut seen = Vec::new();
        while let Some(frame) = inbound.data.next().await {
            match frame {
                DataFrame::SpeechStarted => seen.push(Seen::Started),
                DataFrame::Audio(c) => seen.push(Seen::Audio(c.samples.len())),
                _ => {}
            }
        }
        assert_eq!(
            seen,
            vec![Seen::Started, Seen::Audio(13), Seen::Audio(14)],
            "the cleared ring holds only chunk 13; chunks 11 and 12 were dropped by the interrupt",
        );
    });
}
