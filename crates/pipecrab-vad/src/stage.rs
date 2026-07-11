//! [`VadStage`]: the generic adapter from any [`VoiceActivityDetector`] to a
//! pipeline [`Stage`].

use std::sync::Mutex;

use async_trait::async_trait;
use pipecrab_core::{AudioChunk, DataFrame, Decision, Processor};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{VadError, VoiceActivityDetector};

/// Adapts any [`VoiceActivityDetector`] into a pipeline [`Stage`].
///
/// VAD is a *tap*, not a transform: on a [`DataFrame::Audio`] the stage runs
/// detection **and forwards the audio unchanged**, so a downstream STT stage
/// still sees every sample. What it emits is not the per-window verdict — that
/// would flood the pipeline — but the two *edges* of speech: a
/// [`DataFrame::SpeechStarted`] on the silence→speech transition and a
/// [`DataFrame::SpeechStopped`] on the speech→silence transition. Both ride the
/// **data lane**, in order right behind the audio that produced them, so a
/// downstream stage sees the onset audio before the edge announcing it; a
/// system-lane edge would preempt that audio instead. Between edges it emits
/// nothing.
///
/// The edge is debounced with a hangover ([`VadConfig`]): a run of consecutive
/// windows must agree before the state flips, so a single stray window does not
/// chatter the pipeline with start/stop pairs.
///
/// # Where the state lives
///
/// Edge detection needs one bit — "are we currently in speech?" — to persist
/// across calls, and that bit only becomes known *after* the async
/// [`detect`](VoiceActivityDetector::detect) resolves, inside
/// [`perform`](Stage::perform). Since `perform` takes `&self`, the bit lives
/// behind a [`Mutex`] rather than in the `&mut self` `decide_data`. This stays
/// cancellation-safe: the state is touched only in a single synchronous
/// critical section *after* the `await`, so an [`Interrupt`](pipecrab_core::SystemFrame::Interrupt)
/// that drops an in-flight `perform` (before `detect` returns) changes nothing —
/// there is no torn state to leave behind.
pub struct VadStage<V: VoiceActivityDetector> {
    detector: V,
    config: VadConfig,
    state: Mutex<VadState>,
}

/// Debounce thresholds for [`VadStage`]'s edge detection, both counted in
/// [`detect`](VoiceActivityDetector::detect) windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadConfig {
    /// Consecutive speech windows required to declare speech *started*. Small so
    /// onset is responsive.
    pub start_windows: u32,
    /// Consecutive non-speech windows required to declare speech *stopped*.
    /// Larger, so a brief pause mid-utterance does not clip it.
    pub stop_windows: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        // React to onset immediately; ride out short gaps before closing.
        Self { start_windows: 1, stop_windows: 8 }
    }
}

/// The current run of the edge detector: are we in speech, and how many
/// consecutive windows have disagreed with that so far.
struct VadState {
    in_speech: bool,
    against: u32,
}

/// A speech-state transition worth announcing to the pipeline.
enum Edge {
    Started,
    Stopped,
}

impl VadState {
    /// Feed one window's `is_speech` verdict; return an [`Edge`] if it flips the
    /// state once the [`VadConfig`] hangover is satisfied.
    fn observe(&mut self, is_speech: bool, config: &VadConfig) -> Option<Edge> {
        if is_speech == self.in_speech {
            // Verdict agrees with the current state: reset the opposing run.
            self.against = 0;
            return None;
        }
        self.against += 1;
        let needed = if self.in_speech { config.stop_windows } else { config.start_windows };
        if self.against < needed {
            return None;
        }
        self.in_speech = is_speech;
        self.against = 0;
        Some(if is_speech { Edge::Started } else { Edge::Stopped })
    }
}

impl<V: VoiceActivityDetector> VadStage<V> {
    /// Wrap `detector` as a stage with the default [`VadConfig`].
    pub fn new(detector: V) -> Self {
        Self::with_config(detector, VadConfig::default())
    }

    /// Wrap `detector` as a stage with an explicit [`VadConfig`].
    pub fn with_config(detector: V, config: VadConfig) -> Self {
        Self { detector, config, state: Mutex::new(VadState { in_speech: false, against: 0 }) }
    }
}

/// One audio chunk to run detection over: [`VadStage`]'s [`Processor::Effect`].
/// Emitted by `decide_data`, interpreted by `perform`. Its inner chunk is
/// private — only the stage constructs one.
pub struct Detect(AudioChunk);

impl<V: VoiceActivityDetector> Processor for VadStage<V> {
    type Effect = Detect;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<Detect> {
        match frame {
            // Tap: the audio flows on downstream *and* we run detection over it.
            // The chunk is Arc-backed, so this clone is a refcount bump.
            DataFrame::Audio(chunk) => Decision::forward().emit(Detect(chunk.clone())),
            // Everything else is not ours to inspect.
            _ => Decision::forward(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<V: VoiceActivityDetector> Stage for VadStage<V> {
    async fn perform(&self, Detect(chunk): Detect, out: &Outbound) -> Result<(), StageError> {
        let verdict = self.detector.detect(&chunk.samples, chunk.format).await?;
        // The one place VadStage mutates its own state; see the type doc for why
        // this stays cancellation-safe. The lock is uncontended (one `perform`
        // runs at a time) and held only for the synchronous edge computation.
        let edge = {
            let mut state = self.state.lock().expect("VAD state mutex poisoned");
            state.observe(verdict.is_speech, &self.config)
        };
        let frame = match edge {
            Some(Edge::Started) => DataFrame::SpeechStarted,
            Some(Edge::Stopped) => DataFrame::SpeechStopped,
            None => return Ok(()),
        };
        // The edge rides the data lane, in order behind the audio this same tap
        // already forwarded, so a downstream stage sees the onset before the edge.
        // Ignore the send error: it only happens once the sink has gone away
        // during shutdown, matching the runtime's own forward path.
        let _ = out.send_data(frame).await;
        Ok(())
    }
}

impl From<VadError> for StageError {
    fn from(e: VadError) -> Self {
        // A failed detection is recoverable: skip this window and keep the
        // pipeline alive. The run loop surfaces it as an Error frame upstream.
        StageError::new(e.to_string())
    }
}
