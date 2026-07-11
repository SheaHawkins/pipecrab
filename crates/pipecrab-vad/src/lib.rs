//! pipecrab-vad: the voice-activity-detection interface, in two tiers.
//!
//! Voice activity has two natural shapes, and this crate names both:
//!
//! * [`VoiceActivityDetector`] — the **stage-facing** capability: audio in,
//!   speech *edges* out ([`VadEvent::SpeechStarted`] /
//!   [`VadEvent::SpeechStopped`]). Segmenter-class engines that already speak in
//!   segments — sherpa's VAD, platform VADs — implement this directly. It is
//!   what [`VadStage`] drives.
//! * [`SpeechScorer`] — the **raw-model** tier: a per-window speech
//!   *probability*. This is what a bare silero build (native `ort`, browser
//!   onnxruntime-web) exposes. A scorer becomes a detector through the
//!   [`Debounced`] adapter, which owns the windowing, threshold, and hangover.
//!
//! Probabilities used to live on the one VAD trait; they moved *down* to
//! [`SpeechScorer`], the tier where they are real. A segmenter never has a
//! probability to hand back, and anything downstream that wants confidence
//! (a turn manager, prosody) composes its own [`SpeechScorer`] rather than
//! leaning on the VAD to surface one.
//!
//! # The edge contract
//!
//! Across the lifetime of a [`VoiceActivityDetector`], events **alternate**,
//! starting with [`SpeechStarted`](VadEvent::SpeechStarted): started, stopped,
//! started, stopped, … This is a documented invariant that [`VadStage`] and
//! everything downstream of it trust.
//!
//! [`VadStage`] turns those edges into a **gate**: it owns a pre-roll ring and
//! emits speech-only audio, bracketed by the edges. Note the contract this
//! *inverts* — downstream of the gate, `SpeechStarted` **precedes** an
//! utterance's audio (pre-roll included) and `SpeechStopped` **follows** its
//! last chunk. (The older lane-discipline design emitted the edge *after* the
//! chunk that triggered it; that wording is gone.) See [`VadStage`] for the full
//! gate algorithm.
//!
//! # No runtime format detection
//!
//! Both traits take a bare `&[f32]`, which carries no sample rate. Samples are
//! interpreted as [`input_format()`](VoiceActivityDetector::input_format); no
//! runtime detection is possible. The stage enforces the format *fatally* before
//! any audio reaches an engine, so an engine never sees nonconforming samples
//! through the pipeline — which is why [`VadError`] carries no format-mismatch
//! variant.
//!
//! Platform-neutral and `wasm32`-checkable: the concrete engines live elsewhere,
//! each behind these traits, so the interface itself carries no backend
//! dependency and compiles for the host and `wasm32-unknown-unknown`.
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

/// The stage-facing voice-activity capability: audio in, speech edges out.
///
/// Segmenter-class engines (sherpa's VAD, platform VADs) implement this
/// directly; raw per-window scorers are lifted into it by [`Debounced`]. It is
/// the trait [`VadStage`] drives.
///
/// Like every other pipecrab interface it takes `&self`, so an in-flight call
/// can be dropped on a barge-in interrupt without tearing state. `?Send` on
/// `wasm32` matches pipecrab's single-threaded execution model, so one
/// implementation runs unchanged on a current-thread executor and in the
/// browser, where `Send` bounds cannot be satisfied.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait VoiceActivityDetector: MaybeSendSync {
    /// The one format this detector accepts. The stage caches it and enforces it
    /// *fatally*: samples are interpreted as this format, and no runtime
    /// detection is possible, so nonconforming audio is rejected before it ever
    /// reaches [`process`](Self::process).
    fn input_format(&self) -> AudioFormat;

    /// Feed samples (any length; the engine buffers internally) and return zero
    /// or more edges, in order.
    ///
    /// Across calls, events **must alternate**, starting with
    /// [`SpeechStarted`](VadEvent::SpeechStarted) — the documented invariant
    /// [`VadStage`] and everything downstream trust. Samples are interpreted as
    /// [`input_format()`](Self::input_format).
    async fn process(&self, samples: &[f32]) -> Result<Vec<VadEvent>, VadError>;

    /// Control call: return to the idle, no-speech state. Synchronous,
    /// non-blocking, idempotent (see the [`Processor`](pipecrab_core::Processor)
    /// control-call carve-out). Invoked on an
    /// [`Interrupt`](pipecrab_core::SystemFrame::Interrupt) and by tests.
    ///
    /// There is deliberately no `cancel`: a process quantum is ~1 ms, so there is
    /// nothing worth aborting mid-flight, and structural drop already covers the
    /// await. `reset` is the only control call.
    fn reset(&self);
}

/// The raw-model tier: per-window speech probability.
///
/// This is what a bare silero build implements — the web onnxruntime-web build
/// and any future raw model. A scorer becomes a [`VoiceActivityDetector`]
/// through [`Debounced`], which owns the windowing, threshold, and hangover.
///
/// `?Send` on `wasm32` matches pipecrab's single-threaded execution model, as on
/// [`VoiceActivityDetector`].
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait SpeechScorer: MaybeSendSync {
    /// The one format this scorer accepts. Samples are interpreted as this
    /// format; no runtime detection is possible.
    fn input_format(&self) -> AudioFormat;

    /// Exact window length in samples (e.g. 512 at 16 kHz for silero). Every
    /// call to [`score`](Self::score) is handed exactly this many samples.
    fn window_len(&self) -> usize;

    /// Score exactly [`window_len()`](Self::window_len) samples; returns a
    /// probability in `[0.0, 1.0]` that the window contains speech.
    async fn score(&self, window: &[f32]) -> Result<f32, VadError>;
}

/// Why a [`VoiceActivityDetector::process`] or [`SpeechScorer::score`] call
/// failed.
///
/// Mirrors the message-plus-kind shape of the pipeline's other error types
/// (e.g. `pipecrab-stt`'s `SttError`) so the conversion at a stage boundary is
/// direct.
///
/// There is no format-mismatch variant: [`input_format`](VoiceActivityDetector::input_format)
/// is declared and the stage enforces it fatally, so an engine never sees
/// nonconforming audio, and a bare `&[f32]` carries no rate to validate against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadError {
    /// The engine itself failed — an inference error, a worker that crashed, a
    /// model that never loaded. Carries a human-readable description.
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
