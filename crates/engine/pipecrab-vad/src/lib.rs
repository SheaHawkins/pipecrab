//! Voice-activity detection in two tiers.
//!
//! [`VoiceActivityDetector`] is the stage-facing interface and emits speech
//! edges. [`SpeechScorer`] is the raw-model interface and returns a probability
//! per window. [`Debounced`] adapts a scorer by adding windowing, a threshold,
//! and hangover.
//!
//! # The edge contract
//!
//! Detector events alternate, starting with
//! [`VadEvent::SpeechStarted`]. [`VadStage`] and downstream stages rely on this.
//!
//! [`VadStage`] gates silence and emits each utterance as `SpeechStarted`, its
//! audio including pre-roll, then `SpeechStopped`.
//!
//! # No runtime format detection
//!
//! A sample slice carries no format metadata. [`VadStage`] therefore enforces
//! [`VoiceActivityDetector::input_format`] before calling the engine.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod debounced;
mod stage;

pub use debounced::{DebounceConfig, Debounced};
pub use stage::{GateConfig, VadEffect, VadStage};

use async_trait::async_trait;
use pipecrab_core::AudioFormat;
use pipecrab_runtime::MaybeSendSync;

/// A speech-state transition, emitted by a [`VoiceActivityDetector`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadEvent {
    /// The silence→speech edge: the user started speaking.
    SpeechStarted,
    /// The speech→silence edge: the user stopped speaking.
    SpeechStopped,
}

/// Converts audio into ordered speech edges.
///
/// Segmenting engines implement this directly. [`Debounced`] adapts raw
/// [`SpeechScorer`] implementations for use by [`VadStage`].
///
/// Methods use `&self` so stage effects remain safe to cancel.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait VoiceActivityDetector: MaybeSendSync {
    /// The format this detector accepts.
    fn input_format(&self) -> AudioFormat;

    /// Processes any number of samples and returns ordered edges.
    ///
    /// Across calls, events must alternate starting with
    /// [`VadEvent::SpeechStarted`]. Samples use [`Self::input_format`].
    async fn process(&self, samples: &[f32]) -> Result<Vec<VadEvent>, VadError>;

    /// Returns to the idle state synchronously and idempotently.
    ///
    /// [`VadStage`] calls this after an interrupt. An in-flight `process` future
    /// is cancelled by dropping it.
    fn reset(&self);
}

/// Scores fixed-size audio windows for speech probability.
///
/// Use [`Debounced`] to adapt a scorer into a [`VoiceActivityDetector`].
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait SpeechScorer: MaybeSendSync {
    /// The format this scorer accepts.
    fn input_format(&self) -> AudioFormat;

    /// The exact sample count required by [`Self::score`].
    fn window_len(&self) -> usize;

    /// Score exactly [`window_len()`](Self::window_len) samples; returns a
    /// probability in `[0.0, 1.0]` that the window contains speech.
    async fn score(&self, window: &[f32]) -> Result<f32, VadError>;
}

/// An error from voice-activity processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadError {
    /// A detector or scorer engine failure.
    Engine(String),
}

impl std::fmt::Display for VadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VadError::Engine(msg) => write!(f, "vad engine error: {msg}"),
        }
    }
}

impl std::error::Error for VadError {}
