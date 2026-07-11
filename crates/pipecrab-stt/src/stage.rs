//! [`SttStage`]: the edge-gated adapter from any [`StreamingTranscriber`] to a
//! pipeline [`Stage`], with the pre-roll ring that keeps an utterance's onset.

use std::collections::VecDeque;
use std::time::Duration;

use async_trait::async_trait;
use pipecrab_core::{
    AudioChunk, AudioFormat, DataFrame, Decision, Direction, Processor, SystemFrame, Transcript,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{SttError, SttEvent, StreamingTranscriber};

/// Adapts any [`StreamingTranscriber`] into a pipeline [`Stage`], driven by the
/// VAD's speech edges rather than one utterance per audio frame.
///
/// The stage does not decide *where* an utterance begins and ends — an upstream
/// [`VadStage`](https://docs.rs/pipecrab-vad) does, emitting
/// [`SpeechStarted`](DataFrame::SpeechStarted) /
/// [`SpeechStopped`](DataFrame::SpeechStopped) edges. These ride the **data
/// lane**, so they arrive here in order with the audio and are handled in
/// [`decide_data`]. Between those edges the stage streams live audio into the
/// engine and forwards its [`SttEvent`]s downstream as [`Transcript`]s; outside
/// them it stays idle.
///
/// # The pre-roll ring
///
/// A VAD only declares speech *started* after a run of speech windows
/// ([`start_windows`](https://docs.rs/pipecrab-vad)), so real onset audio has
/// already flowed past by the time the edge arrives. Without pre-roll every
/// utterance would lose its first syllables. So while idle the stage stashes
/// each incoming [`Audio`](DataFrame::Audio) chunk in a duration-bounded
/// `PrerollRing`; on [`SpeechStarted`](DataFrame::SpeechStarted) it drains the
/// ring, in arrival order, into the engine ahead of any live audio. The budget
/// is [`SttConfig::preroll`] (default 300 ms).
///
/// Because the edge travels the data lane *behind* the audio that triggered it,
/// the ring always holds that onset window when `SpeechStarted` arrives — so the
/// stage can always open the utterance from the ring's format, with no special
/// cold-start case.
///
/// # State and the decide/perform split
///
/// Following the [`Processor`]/[`Stage`] split, all state — the ring and the
/// single `in_speech` bit, which is true exactly when the engine has an open
/// utterance — lives in the synchronous `&mut self`
/// `decide_*`; `perform` takes `&self` and only drives the awaited engine calls.
/// The engine itself is a worker-handle (see [`StreamingTranscriber`]), so a
/// barge-in that drops an in-flight [`feed`](StreamingTranscriber::feed) leaves
/// no torn state, and the [`Interrupt`](SystemFrame::Interrupt) cancel is a
/// *control call* invoked right where the interrupt is decided.
///
/// [`decide_data`]: Processor::decide_data
pub struct SttStage<S: StreamingTranscriber> {
    transcriber: S,
    /// Pre-roll ring; accumulates while idle, drained on `SpeechStarted`.
    preroll: PrerollRing,
    /// Whether we are inside an utterance — between a `SpeechStarted` and its
    /// `SpeechStopped`. True exactly when the engine has an open utterance, so it
    /// gates both the live `Feed`s and the closing `End`.
    in_speech: bool,
}

/// Tuning for [`SttStage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SttConfig {
    /// How much onset audio to retain in the pre-roll ring. Larger keeps more of
    /// the utterance's first syllables at the cost of buffering; the default is
    /// 300 ms.
    pub preroll: Duration,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self { preroll: Duration::from_millis(300) }
    }
}

impl<S: StreamingTranscriber> SttStage<S> {
    /// Wrap `transcriber` as a stage with the default [`SttConfig`].
    pub fn new(transcriber: S) -> Self {
        Self::with_config(transcriber, SttConfig::default())
    }

    /// Wrap `transcriber` as a stage with an explicit [`SttConfig`].
    pub fn with_config(transcriber: S, config: SttConfig) -> Self {
        Self { transcriber, preroll: PrerollRing::new(config.preroll), in_speech: false }
    }
}

/// A duration-bounded FIFO of audio chunks: the pre-roll buffer that captures an
/// utterance's onset — the audio that arrives *before* the VAD's `SpeechStarted`
/// edge — so the first syllables are not clipped.
///
/// While the stage is idle it accumulates chunks, evicting the oldest whole
/// chunks once the total buffered duration exceeds `budget`. Chunks vary in
/// size, so eviction works in whole chunks rather than samples.
struct PrerollRing {
    /// The maximum total duration to retain.
    budget: Duration,
    /// Buffered chunks, oldest at the front.
    chunks: VecDeque<AudioChunk>,
}

impl PrerollRing {
    fn new(budget: Duration) -> Self {
        Self { budget, chunks: VecDeque::new() }
    }

    /// The format of the buffered chunks, or `None` when empty. The ring holds a
    /// single uniform format, so the front chunk's format speaks for all.
    fn format(&self) -> Option<AudioFormat> {
        self.chunks.front().map(|c| c.format)
    }

    /// The total buffered duration. Recomputed from the chunks (the ring is
    /// small, ~budget/window) so accounting never drifts.
    fn total(&self) -> Duration {
        self.chunks.iter().map(chunk_duration).sum()
    }

    /// Push a chunk, honouring the format-uniformity and duration-budget rules.
    fn push(&mut self, chunk: AudioChunk) {
        // Format discipline: the ring holds one uniform format. A chunk in a new
        // format means the upstream format changed; discard the stale contents
        // and restart the ring in the new format (resampling is out of scope).
        if let Some(fmt) = self.format() {
            if fmt != chunk.format {
                self.chunks.clear();
            }
        }
        self.chunks.push_back(chunk);
        // Evict oldest whole chunks until we fit the budget, but always keep the
        // most recent chunk: a lone chunk longer than the whole budget is still
        // the freshest onset audio, and dropping it would clip the utterance.
        while self.chunks.len() > 1 && self.total() > self.budget {
            self.chunks.pop_front();
        }
    }

    /// Remove and return every buffered chunk in arrival order.
    fn drain(&mut self) -> Vec<AudioChunk> {
        self.chunks.drain(..).collect()
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

/// One step of the utterance protocol: [`SttStage`]'s [`Processor::Effect`].
/// Emitted by `decide_*`, interpreted by `perform`.
pub enum SttEffect {
    /// Open an utterance in the engine for `format`.
    Begin(AudioFormat),
    /// Feed one window of audio to the open utterance and forward any events.
    Feed(AudioChunk),
    /// Close the utterance, draining its remaining events (including the final).
    End,
}

impl<S: StreamingTranscriber> Processor for SttStage<S> {
    type Effect = SttEffect;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<SttEffect> {
        match frame {
            DataFrame::Audio(chunk) if self.in_speech => {
                // Live speech: consume the audio and feed it to the open utterance.
                // The chunk is Arc-backed, so this clone is a refcount bump.
                Decision::drop().emit(SttEffect::Feed(chunk.clone()))
            }
            DataFrame::Audio(chunk) => {
                // Idle: stash the chunk so the utterance onset — the audio ahead of
                // the VAD's `SpeechStarted` edge — survives. Mutating state in the
                // sync `&mut self` decide step is legal.
                self.preroll.push(chunk.clone());
                Decision::drop()
            }
            DataFrame::SpeechStarted => {
                if self.in_speech {
                    // Duplicate edge: the utterance is already open. Idempotent —
                    // forward, no effects, or `begin_utterance` would run twice.
                    return Decision::forward();
                }
                // The VAD emits this edge on the data lane, in order behind the
                // audio that triggered it, so the ring already holds that onset
                // window: drain it, open the utterance from its format, and replay
                // it. The `Begin`/`Feed` effects run on the data path, raced
                // against the sys lane, so a barge-in can still drop them.
                let preroll = self.preroll.drain();
                match preroll.first().map(|c| c.format) {
                    Some(format) => {
                        self.in_speech = true;
                        let mut decision = Decision::forward().emit(SttEffect::Begin(format));
                        for chunk in preroll {
                            decision = decision.emit(SttEffect::Feed(chunk));
                        }
                        decision
                    }
                    None => {
                        // Empty ring: no audio preceded the edge. A VAD never does
                        // this (it fires only after a speech window), so it means a
                        // malformed upstream. Stay idle rather than open an
                        // utterance we have no format for; still forward the edge.
                        Decision::forward()
                    }
                }
            }
            DataFrame::SpeechStopped => {
                if !self.in_speech {
                    // Not in an utterance: nothing to close.
                    return Decision::forward();
                }
                self.in_speech = false;
                Decision::forward().emit(SttEffect::End)
            }
            // Transcripts, transport bytes, custom frames: not ours to touch.
            _ => Decision::forward(),
        }
    }

    fn decide_system(&mut self, _dir: Direction, frame: &SystemFrame) -> Decision<SttEffect> {
        match frame {
            SystemFrame::Interrupt => {
                // Control call (see the `Processor` control-call carve-out): flip
                // the engine's cancel flag right where the interrupt is decided,
                // so the in-flight utterance is abandoned promptly and unmissably.
                // Sync, non-blocking, idempotent, infallible — sound from the
                // `&mut self` decide step.
                self.transcriber.cancel();
                self.in_speech = false;
                // The ring is left intact: it only accumulates while idle, so
                // there is nothing from the cancelled utterance to clear.
                Decision::forward()
            }
            // Start, Stop, Error, and any future frames: pass through untouched.
            _ => Decision::forward(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<S: StreamingTranscriber> Stage for SttStage<S> {
    async fn perform(&self, effect: SttEffect, out: &Outbound) -> Result<(), StageError> {
        match effect {
            SttEffect::Begin(format) => {
                self.transcriber.begin_utterance(format).await?;
            }
            SttEffect::Feed(chunk) => {
                let events = self.transcriber.feed(&chunk.samples).await?;
                forward_events(events, out).await;
            }
            SttEffect::End => {
                let events = self.transcriber.end_utterance().await?;
                forward_events(events, out).await;
            }
        }
        Ok(())
    }
}

/// Forward each engine event downstream as a [`Transcript`]. `Endpoint` is a v1
/// no-op — the engine's own end-of-utterance signal has no frame to map to yet
/// (a future `TurnEnded` is out of scope), so it is ignored.
async fn forward_events(events: Vec<SttEvent>, out: &Outbound) {
    for event in events {
        let transcript = match event {
            SttEvent::Partial { text, stable } => Transcript::user_partial(text, stable),
            SttEvent::Final(text) => Transcript::user_final(text),
            SttEvent::Endpoint => continue,
        };
        // Ignore the send error: it only fires once the sink is gone during
        // shutdown, matching the runtime's own forward path.
        let _ = out.send_data(transcript.into()).await;
    }
}

impl From<SttError> for StageError {
    fn from(e: SttError) -> Self {
        // A failed engine call is recoverable: skip it and keep the pipeline
        // alive. The run loop surfaces it as an Error frame upstream.
        StageError::new(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn chunk(sample_rate: u32, channels: u16, samples: usize) -> AudioChunk {
        AudioChunk::new(Arc::from(vec![0.0f32; samples]), AudioFormat::new(sample_rate, channels))
    }

    #[test]
    fn chunk_duration_is_frames_over_sample_rate() {
        // 16 000 mono samples at 16 kHz is exactly one second.
        assert_eq!(chunk_duration(&chunk(16_000, 1, 16_000)), Duration::from_secs(1));
        // 1 kHz mono makes one sample == one millisecond.
        assert_eq!(chunk_duration(&chunk(1_000, 1, 250)), Duration::from_millis(250));
    }

    #[test]
    fn chunk_duration_counts_interleaved_frames_not_samples() {
        // Stereo: 480 interleaved samples is 240 frames, so 240/48k = 5 ms — half
        // what a naive samples/rate would give.
        assert_eq!(chunk_duration(&chunk(48_000, 2, 480)), Duration::from_millis(5));
    }

    #[test]
    fn chunk_duration_of_empty_or_degenerate_is_zero() {
        assert_eq!(chunk_duration(&chunk(16_000, 1, 0)), Duration::ZERO);
        // A zero sample rate can't yield a duration; guard rather than divide by it.
        assert_eq!(chunk_duration(&chunk(0, 1, 100)), Duration::ZERO);
    }
}
