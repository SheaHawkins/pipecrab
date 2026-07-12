//! Gates audio using a [`VoiceActivityDetector`].

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use pipecrab_core::{
    AudioChunk, AudioFormat, DataFrame, Decision, Direction, Processor, SystemFrame,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{VadError, VadEvent, VoiceActivityDetector};

/// Emits speech-only audio bracketed by speech edges.
///
/// Each utterance is [`DataFrame::SpeechStarted`], its audio including pre-roll,
/// then [`DataFrame::SpeechStopped`]. Silence is dropped.
///
/// Downstream stages may rely on edges bracketing all utterance audio.
///
/// # The pre-roll ring
///
/// Detection lags speech onset, so the gate buffers idle audio for
/// [`GateConfig::preroll`] and flushes it when speech starts.
///
/// # Topology commitment
///
/// Consumers that need continuous audio must run upstream of this stage.
///
/// # State and cancellation
///
/// Gate state changes in one synchronous critical section after
/// [`VoiceActivityDetector::process`]. Sends occur after releasing the lock, so
/// cancellation cannot carry stale buffered audio into another utterance.
pub struct VadStage<V: VoiceActivityDetector> {
    detector: V,
    /// The cached [`VoiceActivityDetector::input_format`].
    expected: AudioFormat,
    /// Gate mode and pre-roll buffer.
    gate: Mutex<Gate>,
}

/// Tuning for [`VadStage`]'s pre-roll ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateConfig {
    /// Maximum onset audio retained before speech starts.
    pub preroll: Duration,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            preroll: Duration::from_millis(300),
        }
    }
}

/// The gate's mutable state: whether we are passing speech through, and the
/// pre-roll ring that accumulates while idle.
struct Gate {
    mode: Mode,
    ring: PrerollRing,
}

/// Which side of the gate we are on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// No open utterance: incoming audio accumulates in the ring, nothing is
    /// emitted.
    Idle,
    /// An utterance is open: incoming audio flows through as [`DataFrame::Audio`].
    Speech,
}

/// A duration-bounded FIFO that preserves audio preceding a speech-start edge.
///
/// It evicts whole chunks from the front when the duration exceeds its budget.
/// The stage guarantees a uniform format.
struct PrerollRing {
    /// The maximum total duration to retain.
    budget: Duration,
    /// Buffered chunks, oldest at the front.
    chunks: VecDeque<AudioChunk>,
}

impl PrerollRing {
    fn new(budget: Duration) -> Self {
        Self {
            budget,
            chunks: VecDeque::new(),
        }
    }

    /// Computes the total buffered duration.
    fn total(&self) -> Duration {
        self.chunks.iter().map(chunk_duration).sum()
    }

    /// Push a chunk, evicting oldest whole chunks to honour the duration budget.
    fn push(&mut self, chunk: AudioChunk) {
        self.chunks.push_back(chunk);
        // Evict oldest whole chunks until we fit the budget, but always keep the
        // most recent chunk: a lone chunk longer than the whole budget is still
        // the freshest onset audio, and dropping it would clip the utterance.
        while self.chunks.len() > 1 && self.total() > self.budget {
            self.chunks.pop_front();
        }
    }

    /// Remove and return every buffered chunk in arrival order.
    fn take(&mut self) -> Vec<AudioChunk> {
        self.chunks.drain(..).collect()
    }

    /// Discard every buffered chunk.
    fn clear(&mut self) {
        self.chunks.clear();
    }
}

/// The wall-clock duration of one audio chunk: interleaved frames over the
/// sample rate. A malformed format (zero rate) yields zero rather than dividing
/// by it.
fn chunk_duration(chunk: &AudioChunk) -> Duration {
    let channels = chunk.format.channels.max(1) as u64;
    let rate = chunk.format.sample_rate as u64;
    if rate == 0 {
        return Duration::ZERO;
    }
    let frames = chunk.samples.len() as u64 / channels;
    // Integer nanoseconds keep the budget accounting exact and drift-free.
    Duration::from_nanos(frames * 1_000_000_000 / rate)
}

impl<V: VoiceActivityDetector> VadStage<V> {
    /// Wrap `detector` as a gate with the default [`GateConfig`].
    pub fn new(detector: V) -> Self {
        Self::with_config(detector, GateConfig::default())
    }

    /// Wrap `detector` as a gate with an explicit [`GateConfig`].
    pub fn with_config(detector: V, config: GateConfig) -> Self {
        let expected = detector.input_format();
        Self {
            detector,
            expected,
            gate: Mutex::new(Gate {
                mode: Mode::Idle,
                ring: PrerollRing::new(config.preroll),
            }),
        }
    }
}

/// One operation for [`VadStage`] to perform.
pub struct VadEffect(Effect);

enum Effect {
    /// Run detection over this conforming chunk and drive the gate.
    Detect(AudioChunk),
    /// The chunk's format did not match the detector's; fail fatally.
    RejectFormat { got: AudioFormat },
}

impl<V: VoiceActivityDetector> Processor for VadStage<V> {
    type Effect = VadEffect;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<VadEffect> {
        match frame {
            // Format-fatal admission: a mismatch is rejected before any audio
            // reaches the engine (the engine cannot detect rate from `&[f32]`).
            DataFrame::Audio(chunk) if chunk.format != self.expected => {
                Decision::drop().emit(VadEffect(Effect::RejectFormat { got: chunk.format }))
            }
            // Conforming audio: drop it — the gate owns forwarding now — and let
            // `perform` decide whether it is gated through or stashed in the ring.
            // The chunk is Arc-backed, so this clone is a refcount bump.
            DataFrame::Audio(chunk) => {
                Decision::drop().emit(VadEffect(Effect::Detect(chunk.clone())))
            }
            // Everything else is not ours to inspect.
            _ => Decision::forward(),
        }
    }

    fn decide_system(&mut self, _dir: Direction, frame: &SystemFrame) -> Decision<VadEffect> {
        match frame {
            SystemFrame::Interrupt => {
                // An Interrupt reaching VadStage is a head-injected session
                // abandon — turn-manager interrupts originate downstream and never
                // travel up here — so gate/downstream protocol coherence across it
                // is explicitly not guaranteed. Both sides re-sync to idle: the
                // gate resets, and SttStage cancels on the same Interrupt.
                {
                    let mut gate = self.gate.lock().expect("VAD gate mutex poisoned");
                    gate.mode = Mode::Idle;
                    gate.ring.clear();
                }
                // Control call: return the detector to its idle, no-speech state.
                self.detector.reset();
                Decision::forward()
            }
            // Start, Stop, Error, and any future frames: pass through untouched.
            _ => Decision::forward(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<V: VoiceActivityDetector> Stage for VadStage<V> {
    async fn perform(&self, effect: VadEffect, out: &Outbound) -> Result<(), StageError> {
        let chunk = match effect.0 {
            Effect::RejectFormat { got } => {
                return Err(StageError::fatal(format!(
                    "VadStage requires {} Hz/{} ch (declared by the engine); \
                     got {} Hz/{} ch — insert a resample stage upstream or \
                     reconfigure the source",
                    self.expected.sample_rate,
                    self.expected.channels,
                    got.sample_rate,
                    got.channels,
                )));
            }
            Effect::Detect(chunk) => chunk,
        };

        let events = self.detector.process(&chunk.samples).await?;

        // ONE critical section: take the ring, flip the mode, and build the send
        // plan atomically. No await inside; the sends happen after the unlock.
        let plan: Vec<DataFrame> = {
            let mut gate = self.gate.lock().expect("VAD gate mutex poisoned");
            let mut plan = Vec::new();
            let mut sent_chunk = false;
            for event in &events {
                match event {
                    VadEvent::SpeechStarted => {
                        // Alternation invariant: a SpeechStarted while already in
                        // speech is a misbehaving detector. Surface it loudly in
                        // debug; in release, re-opening from Idle mode below still
                        // produces coherent, bracketed output.
                        debug_assert!(
                            gate.mode == Mode::Idle,
                            "VAD alternation violated: SpeechStarted while already in speech",
                        );
                        plan.push(DataFrame::SpeechStarted);
                        // The whole onset window survives: the ring's chunks in
                        // arrival order, then the triggering chunk.
                        for pre in gate.ring.take() {
                            plan.push(DataFrame::Audio(pre));
                        }
                        plan.push(DataFrame::Audio(chunk.clone()));
                        sent_chunk = true;
                        gate.mode = Mode::Speech;
                    }
                    VadEvent::SpeechStopped => {
                        debug_assert!(
                            gate.mode == Mode::Speech,
                            "VAD alternation violated: SpeechStopped while not in speech",
                        );
                        // The tail chunk closes the utterance, then the edge —
                        // unless a Started in this same batch already sent it.
                        if !sent_chunk {
                            plan.push(DataFrame::Audio(chunk.clone()));
                            sent_chunk = true;
                        }
                        plan.push(DataFrame::SpeechStopped);
                        gate.mode = Mode::Idle;
                    }
                }
            }
            if events.is_empty() {
                match gate.mode {
                    // Live speech: the chunk flows straight through.
                    Mode::Speech => plan.push(DataFrame::Audio(chunk.clone())),
                    // Idle silence: accumulate it in the ring, emit nothing.
                    Mode::Idle => gate.ring.push(chunk.clone()),
                }
            }
            plan
        };

        for frame in plan {
            // Ignore the send error: it only fires once the sink has gone away
            // during shutdown, matching the runtime's own forward path.
            let _ = out.send_data(frame).await;
        }
        Ok(())
    }
}

impl From<VadError> for StageError {
    fn from(e: VadError) -> Self {
        // A failed detection is recoverable: skip this chunk and keep the
        // pipeline alive. Only the format path (RejectFormat) is fatal.
        StageError::new(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn chunk(sample_rate: u32, channels: u16, samples: usize) -> AudioChunk {
        AudioChunk::new(
            Arc::from(vec![0.0f32; samples]),
            AudioFormat::new(sample_rate, channels),
        )
    }

    #[test]
    fn chunk_duration_is_frames_over_sample_rate() {
        // 16 000 mono samples at 16 kHz is exactly one second.
        assert_eq!(
            chunk_duration(&chunk(16_000, 1, 16_000)),
            Duration::from_secs(1)
        );
        // 1 kHz mono makes one sample == one millisecond.
        assert_eq!(
            chunk_duration(&chunk(1_000, 1, 250)),
            Duration::from_millis(250)
        );
    }

    #[test]
    fn chunk_duration_counts_interleaved_frames_not_samples() {
        // Stereo: 480 interleaved samples is 240 frames, so 240/48k = 5 ms — half
        // what a naive samples/rate would give.
        assert_eq!(
            chunk_duration(&chunk(48_000, 2, 480)),
            Duration::from_millis(5)
        );
    }

    #[test]
    fn chunk_duration_of_empty_or_degenerate_is_zero() {
        assert_eq!(chunk_duration(&chunk(16_000, 1, 0)), Duration::ZERO);
        // A zero sample rate can't yield a duration; guard rather than divide by it.
        assert_eq!(chunk_duration(&chunk(0, 1, 100)), Duration::ZERO);
    }

    #[test]
    fn preroll_evicts_by_duration_keeping_the_most_recent() {
        // 1000 Hz mono makes 1 sample == 1 ms, so a 100 ms budget holds ~100 samples.
        let mut ring = PrerollRing::new(Duration::from_millis(100));
        //   +20        -> [20]        (20 ms)
        //   +50        -> [20,50]     (70 ms)
        //   +40 (110)  -> evict 20    -> [50,40]    (90 ms)
        //   +30 (120)  -> evict 50    -> [40,30]    (70 ms)
        for n in [20usize, 50, 40, 30] {
            ring.push(chunk(1000, 1, n));
        }
        let survivors: Vec<usize> = ring.take().iter().map(|c| c.samples.len()).collect();
        assert_eq!(
            survivors,
            vec![40, 30],
            "only the last two chunks survive, in arrival order"
        );
    }

    #[test]
    fn preroll_keeps_a_lone_oversized_chunk() {
        // A single chunk longer than the whole budget is still the freshest onset
        // audio: keep it rather than clip the utterance.
        let mut ring = PrerollRing::new(Duration::from_millis(10));
        ring.push(chunk(1000, 1, 500)); // 500 ms >> 10 ms budget
        let survivors: Vec<usize> = ring.take().iter().map(|c| c.samples.len()).collect();
        assert_eq!(
            survivors,
            vec![500],
            "the most-recent chunk is never evicted"
        );
    }
}
