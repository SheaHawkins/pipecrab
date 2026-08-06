//! End-to-end barge-in: a full VAD → STT → BargeIn → LM → chunker → TTS
//! pipeline crossing real Interrupts.
//!
//! The mocks are cribbed from each crate's own stage tests: a scripted VAD, a
//! counting streaming transcriber, a language model whose first generation
//! parks mid-stream (so a barge-in can drop it), and a synthesizer that stamps
//! each chunk with its call number so the output audio is attributable to a
//! generation.
//!
//! Deterministic and tokio-free (`block_on`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::executor::block_on;
use futures::sink::SinkExt;
use futures::stream::StreamExt;
use pipecrab_core::{AudioChunk, AudioFormat, DataFrame, Direction, SystemFrame};
use pipecrab_lm::{
    Conversation, GenParams, LanguageModel, LmError, LmStage, ModelDelta, ModelStream,
    ToolDefinition,
};
use pipecrab_runtime::{PipelineBuilder, Received};
use pipecrab_stt::{StreamingTranscriber, SttError, SttEvent, SttStage};
use pipecrab_tts::{SentenceChunker, Synthesizer, TtsAudioStream, TtsError, TtsStage};
use pipecrab_turn::BargeInStage;
use pipecrab_vad::{VadError, VadEvent, VadStage, VoiceActivityDetector};

const FMT: AudioFormat = AudioFormat {
    sample_rate: 16_000,
    channels: 1,
};

fn audio(n: usize) -> DataFrame {
    DataFrame::Audio(AudioChunk::new(Arc::from(vec![0.0f32; n]), FMT))
}

// --- Scripted VAD: one edge-batch per process call. ---------------------------

struct ScriptedVad {
    script: Mutex<VecDeque<Vec<VadEvent>>>,
    resets: Arc<AtomicUsize>,
}

impl ScriptedVad {
    fn new(script: Vec<Vec<VadEvent>>) -> (Self, Arc<AtomicUsize>) {
        let resets = Arc::new(AtomicUsize::new(0));
        (
            Self {
                script: Mutex::new(script.into_iter().collect()),
                resets: resets.clone(),
            },
            resets,
        )
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl VoiceActivityDetector for ScriptedVad {
    fn input_format(&self) -> AudioFormat {
        FMT
    }

    async fn process(&self, _samples: Arc<[f32]>) -> Result<Vec<VadEvent>, VadError> {
        Ok(self.script.lock().unwrap().pop_front().unwrap_or_default())
    }

    fn reset(&self) {
        self.resets.fetch_add(1, Ordering::SeqCst);
    }
}

// --- Counting STT: a distinct final per utterance. ----------------------------

#[derive(Default)]
struct SttProbe {
    begins: AtomicUsize,
    ends: AtomicUsize,
    cancels: AtomicUsize,
}

struct CountingStt {
    probe: Arc<SttProbe>,
}

impl CountingStt {
    fn new() -> (Self, Arc<SttProbe>) {
        let probe = Arc::new(SttProbe::default());
        (
            Self {
                probe: probe.clone(),
            },
            probe,
        )
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl StreamingTranscriber for CountingStt {
    fn input_format(&self) -> AudioFormat {
        FMT
    }

    async fn begin_utterance(&self) -> Result<(), SttError> {
        self.probe.begins.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn feed(&self, _samples: Arc<[f32]>) -> Result<Vec<SttEvent>, SttError> {
        Ok(Vec::new())
    }

    async fn end_utterance(&self) -> Result<Vec<SttEvent>, SttError> {
        let n = self.probe.ends.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(vec![SttEvent::Final(format!("utterance {n}").into())])
    }

    fn cancel(&self) {
        self.probe.cancels.fetch_add(1, Ordering::SeqCst);
    }
}

// --- A language model whose first generation parks mid-stream. ----------------

#[derive(Default)]
struct LmProbe {
    generations: AtomicUsize,
    cancels: AtomicUsize,
}

struct SmokeLm {
    /// Per-generation delta scripts, consumed in call order.
    scripts: Mutex<VecDeque<Vec<ModelDelta>>>,
    /// Present until the first generation consumes it: after its deltas, the
    /// stream signals `reached` and parks on `block` until dropped.
    park: Mutex<Option<(mpsc::Sender<()>, oneshot::Receiver<()>)>>,
    probe: Arc<LmProbe>,
}

impl SmokeLm {
    fn new(
        scripts: Vec<Vec<ModelDelta>>,
        park: Option<(mpsc::Sender<()>, oneshot::Receiver<()>)>,
    ) -> (Self, Arc<LmProbe>) {
        let probe = Arc::new(LmProbe::default());
        (
            Self {
                scripts: Mutex::new(scripts.into_iter().collect()),
                park: Mutex::new(park),
                probe: probe.clone(),
            },
            probe,
        )
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl LanguageModel for SmokeLm {
    async fn generate(
        &self,
        _convo: &Conversation,
        _params: &GenParams,
        _tools: &[ToolDefinition],
    ) -> Result<ModelStream, LmError> {
        self.probe.generations.fetch_add(1, Ordering::SeqCst);
        let deltas: VecDeque<ModelDelta> = self
            .scripts
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_default()
            .into();
        if let Some((reached, block)) = self.park.lock().unwrap().take() {
            let stream = futures::stream::unfold(
                (deltas, reached, block),
                |(mut deltas, mut reached, block)| async move {
                    match deltas.pop_front() {
                        Some(delta) => Some((Ok(delta), (deltas, reached, block))),
                        None => {
                            // Signal the test, then park: only a barge-in
                            // dropping this future ends the generation.
                            let _ = reached.send(()).await;
                            let _ = block.await;
                            None
                        }
                    }
                },
            );
            Ok(stream.boxed())
        } else {
            let items: Vec<Result<ModelDelta, LmError>> = deltas.into_iter().map(Ok).collect();
            Ok(futures::stream::iter(items).boxed())
        }
    }

    fn cancel(&self) {
        self.probe.cancels.fetch_add(1, Ordering::SeqCst);
    }

    async fn save_state(&self) -> Result<Vec<u8>, LmError> {
        Ok(Vec::new())
    }

    async fn load_state(&self, _blob: &[u8]) -> Result<(), LmError> {
        Ok(())
    }
}

// --- A synthesizer stamping each chunk with its call number. ------------------

#[derive(Default)]
struct SynthProbe {
    calls: AtomicUsize,
    cancels: AtomicUsize,
}

struct StampingSynth {
    probe: Arc<SynthProbe>,
    /// Signalled once per synthesis, after its chunk has gone downstream.
    emitted: mpsc::Sender<()>,
}

impl StampingSynth {
    fn new(emitted: mpsc::Sender<()>) -> (Self, Arc<SynthProbe>) {
        let probe = Arc::new(SynthProbe::default());
        (
            Self {
                probe: probe.clone(),
                emitted,
            },
            probe,
        )
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Synthesizer for StampingSynth {
    fn output_format(&self) -> AudioFormat {
        AudioFormat::new(24_000, 1)
    }

    async fn synthesize(&self, _text: &str) -> Result<TtsAudioStream, TtsError> {
        let n = self.probe.calls.fetch_add(1, Ordering::SeqCst) + 1;
        enum Step {
            Chunk(usize, mpsc::Sender<()>),
            Signal(mpsc::Sender<()>),
        }
        let stream = futures::stream::unfold(Step::Chunk(n, self.emitted.clone()), |step| async {
            match step {
                Step::Chunk(n, emitted) => {
                    let chunk =
                        AudioChunk::new(Arc::from(vec![n as f32]), AudioFormat::new(24_000, 1));
                    Some((Ok(chunk), Step::Signal(emitted)))
                }
                Step::Signal(mut emitted) => {
                    // The chunk is already downstream when the stage polls
                    // again; signal the test, then end the stream.
                    let _ = emitted.send(()).await;
                    None
                }
            }
        });
        Ok(stream.boxed())
    }

    fn cancel(&self) {
        self.probe.cancels.fetch_add(1, Ordering::SeqCst);
    }
}

// --- Harness. -----------------------------------------------------------------

/// A tag per tail event, for compact assertions.
#[derive(Debug, PartialEq)]
enum Seen {
    Started,
    Stopped,
    Interrupt,
    /// A TTS chunk, tagged with its synthesis call number.
    Audio(f32),
}

async fn drain(mut output: pipecrab_runtime::Inbound) -> Vec<Seen> {
    let mut seen = Vec::new();
    while let Some(received) = output.recv().await {
        match received {
            Received::Data(DataFrame::SpeechStarted) => seen.push(Seen::Started),
            Received::Data(DataFrame::SpeechStopped) => seen.push(Seen::Stopped),
            Received::Data(DataFrame::Audio(c)) => seen.push(Seen::Audio(c.samples[0])),
            Received::Sys(_, SystemFrame::Interrupt) => seen.push(Seen::Interrupt),
            _ => {}
        }
    }
    seen
}

fn count(seen: &[Seen], tag: fn(&Seen) -> bool) -> usize {
    seen.iter().filter(|s| tag(s)).count()
}

// --- Tests. -------------------------------------------------------------------

#[test]
fn a_barge_in_stops_the_reply_and_answers_the_interrupting_utterance() {
    block_on(async {
        // Utterance 1 (chunks 1-2) parks the LM mid-reply; utterance 2
        // (chunks 3-4) barges in, and its own transcript must produce a second
        // generation — the regression test for the causal flush.
        let (vad, vad_resets) = ScriptedVad::new(vec![
            vec![VadEvent::SpeechStarted],
            vec![VadEvent::SpeechStopped],
            vec![VadEvent::SpeechStarted],
            vec![VadEvent::SpeechStopped],
        ]);
        let (stt, stt_probe) = CountingStt::new();
        let (lm_reached_tx, mut lm_reached_rx) = mpsc::channel::<()>(1);
        let (lm_block_tx, lm_block_rx) = oneshot::channel::<()>();
        let (lm, lm_probe) = SmokeLm::new(
            vec![
                vec![ModelDelta::Text(Arc::from("first reply. "))],
                vec![ModelDelta::Text(Arc::from("second reply."))],
            ],
            Some((lm_reached_tx, lm_block_rx)),
        );
        let (synth_emitted_tx, mut synth_emitted_rx) = mpsc::channel::<()>(2);
        let (synth, synth_probe) = StampingSynth::new(synth_emitted_tx);

        let (ends, driver) = PipelineBuilder::new()
            .stage(VadStage::new(vad))
            .stage(SttStage::new(stt))
            .stage(BargeInStage::new())
            .stage(LmStage::new(lm, "system prompt"))
            .stage(SentenceChunker::new())
            .stage(TtsStage::new(synth))
            .build()
            .start();
        let input = ends.input;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            // Utterance 1.
            let _ = input.send_data(audio(160)).await;
            let _ = input.send_data(audio(160)).await;
            // Wait for generation 1 to emit its sentence and park, and for the
            // synthesized audio to be downstream, before barging in.
            lm_reached_rx.next().await.expect("generation 1 must park");
            synth_emitted_rx
                .next()
                .await
                .expect("generation 1 audio must be out");
            // Utterance 2: the barge-in.
            let _ = input.send_data(audio(160)).await;
            let _ = input.send_data(audio(160)).await;
            // Returning drops `input`, cascading shutdown; generation 2 runs
            // to completion during the drain.
        };

        let (_, seen, _) = futures::join!(feed, drain(ends.output), driver);

        // Two speech onsets: BargeInStage interrupts on each (the first is an
        // idle no-op) and re-emits both edges.
        assert_eq!(count(&seen, |s| *s == Seen::Started), 2, "{seen:?}");
        assert_eq!(count(&seen, |s| *s == Seen::Stopped), 2, "{seen:?}");
        assert_eq!(count(&seen, |s| *s == Seen::Interrupt), 2, "{seen:?}");

        // The barge-in dropped the parked generation...
        assert!(
            lm_block_tx.is_canceled(),
            "the parked generation must have been dropped"
        );
        assert_eq!(lm_probe.cancels.load(Ordering::SeqCst), 2);
        assert_eq!(synth_probe.cancels.load(Ordering::SeqCst), 2);
        // ...but never reached the stages upstream of the originator.
        assert_eq!(stt_probe.cancels.load(Ordering::SeqCst), 0);
        assert_eq!(vad_resets.load(Ordering::SeqCst), 0);

        // Both utterances were transcribed, and — the causal-flush regression —
        // the barge-in utterance itself produced the second generation.
        assert_eq!(stt_probe.begins.load(Ordering::SeqCst), 2);
        assert_eq!(stt_probe.ends.load(Ordering::SeqCst), 2);
        assert_eq!(lm_probe.generations.load(Ordering::SeqCst), 2);

        // One audio chunk per generation, in order.
        let audio: Vec<f32> = seen
            .iter()
            .filter_map(|s| match s {
                Seen::Audio(v) => Some(*v),
                _ => None,
            })
            .collect();
        assert_eq!(audio, vec![1.0, 2.0], "{seen:?}");
    });
}

#[test]
fn a_head_injected_interrupt_while_idle_is_harmless() {
    block_on(async {
        // Both generations finish normally; between the two utterances the
        // application injects an Interrupt at the head — the session-abandon
        // path. Every stage resets or no-ops, and the pipeline keeps working.
        let (vad, vad_resets) = ScriptedVad::new(vec![
            vec![VadEvent::SpeechStarted],
            vec![VadEvent::SpeechStopped],
            vec![VadEvent::SpeechStarted],
            vec![VadEvent::SpeechStopped],
        ]);
        let (stt, stt_probe) = CountingStt::new();
        let (lm, lm_probe) = SmokeLm::new(
            vec![
                vec![ModelDelta::Text(Arc::from("first reply."))],
                vec![ModelDelta::Text(Arc::from("second reply."))],
            ],
            None,
        );
        let (synth_emitted_tx, mut synth_emitted_rx) = mpsc::channel::<()>(2);
        let (synth, _synth_probe) = StampingSynth::new(synth_emitted_tx);

        let (ends, driver) = PipelineBuilder::new()
            .stage(VadStage::new(vad))
            .stage(SttStage::new(stt))
            .stage(BargeInStage::new())
            .stage(LmStage::new(lm, "system prompt"))
            .stage(SentenceChunker::new())
            .stage(TtsStage::new(synth))
            .build()
            .start();
        let input = ends.input;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let _ = input.send_data(audio(160)).await;
            let _ = input.send_data(audio(160)).await;
            // Reply 1 is fully out; the pipeline is idle.
            synth_emitted_rx
                .next()
                .await
                .expect("generation 1 audio must be out");
            let _ = input
                .send_system(Direction::Down, SystemFrame::Interrupt)
                .await;
            // A second utterance still works after the session abandon.
            let _ = input.send_data(audio(160)).await;
            let _ = input.send_data(audio(160)).await;
        };

        let (_, seen, _) = futures::join!(feed, drain(ends.output), driver);

        // The head-injected Interrupt reached VAD and STT (unlike a barge-in,
        // which originates below them).
        assert_eq!(vad_resets.load(Ordering::SeqCst), 1);
        assert_eq!(stt_probe.cancels.load(Ordering::SeqCst), 1);

        // Both utterances produced replies.
        assert_eq!(
            lm_probe.generations.load(Ordering::SeqCst),
            2,
            "begins={} ends={} seen={seen:?}",
            stt_probe.begins.load(Ordering::SeqCst),
            stt_probe.ends.load(Ordering::SeqCst),
        );
        let audio: Vec<f32> = seen
            .iter()
            .filter_map(|s| match s {
                Seen::Audio(v) => Some(*v),
                _ => None,
            })
            .collect();
        assert_eq!(audio, vec![1.0, 2.0], "{seen:?}");
    });
}
