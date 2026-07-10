//! pipecrab-vad: the voice-activity-detection interface.
//!
//! One trait — [`VoiceActivityDetector`] — is the swappable VAD capability:
//! `f32` samples in, a [`VadVerdict`] (is this window speech, and how sure?)
//! out. It mirrors [`pipecrab-stt`](https://docs.rs/pipecrab-stt)'s
//! `Transcriber` trait: the concrete engines — a native `ort`-hosted build, the
//! browser onnxruntime-web build — live in their own crates behind this one
//! trait, so the pipeline above never names a model and the interface itself carries
//! no backend dependency. It compiles for the host and for
//! `wasm32-unknown-unknown`.
//!
//! The trait only answers "speech or not, right now." [`VadStage`] layers the
//! segmentation on top: it runs the detector per window and collapses the
//! verdict stream into just the two *edges* of speech
//! ([`SpeechStarted`](pipecrab_core::SystemFrame::SpeechStarted) /
//! [`SpeechStopped`](pipecrab_core::SystemFrame::SpeechStopped)), so the
//! pipeline sees a handful of control frames, not a per-window flood.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod stage;
pub use stage::{Detect, VadConfig, VadStage};

use async_trait::async_trait;
use pipecrab_core::AudioFormat;
use pipecrab_runtime::MaybeSendSync;

/// The swappable voice-activity-detection capability: `f32` samples in, a
/// [`VadVerdict`] out.
///
/// This is the durable interface. A native engine (`ort`-hosted) and a
/// browser engine (onnxruntime-web in a Web Worker) both implement this one
/// trait, so the stage above never names a concrete model. Like every other
/// pipecrab interface it takes `&self`, so an in-flight call can be dropped on a
/// barge-in interrupt without tearing state.
///
/// `?Send` on `wasm32` matches pipecrab's single-threaded execution model, so
/// one implementation runs unchanged on a current-thread executor and in the
/// browser, where `Send` bounds cannot be satisfied.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait VoiceActivityDetector: MaybeSendSync {
    /// Classify `samples` (interleaved `f32` PCM in `format`) as speech or not.
    ///
    /// An engine that accepts only one format (say, 16 kHz mono) should
    /// reject a mismatch with [`VadError::UnsupportedFormat`] rather than
    /// resample — resampling belongs to a separate stage.
    async fn detect(&self, samples: &[f32], format: AudioFormat)
        -> Result<VadVerdict, VadError>;
}

/// The outcome of a single [`VoiceActivityDetector::detect`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadVerdict {
    /// The model's probability, in `[0.0, 1.0]`, that the window contains speech.
    pub speech_probability: f32,
    /// Whether the window counts as speech — the probability thresholded by the
    /// engine's configured cutoff.
    pub is_speech: bool,
}

/// Why a [`VoiceActivityDetector::detect`] call failed.
///
/// Mirrors the message-plus-kind shape of the pipeline's other error types
/// (e.g. `pipecrab-stt`'s `SttError`) so the conversion at a stage boundary is
/// direct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadError {
    /// The detector engine itself failed — an inference error, a worker that
    /// crashed, a model that never loaded. Carries a human-readable description.
    Engine(String),
    /// The detector requires a specific input format and got another. It does
    /// not resample: feed it audio in the format it expects.
    UnsupportedFormat {
        /// The format this detector accepts (e.g. 16 kHz mono).
        expected: AudioFormat,
        /// The format of the rejected samples.
        got: AudioFormat,
    },
}

impl std::fmt::Display for VadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VadError::Engine(msg) => write!(f, "vad engine error: {msg}"),
            VadError::UnsupportedFormat { expected, got } => write!(
                f,
                "vad format mismatch: detector expects {} Hz/{} ch, got {} Hz/{} ch",
                expected.sample_rate, expected.channels, got.sample_rate, got.channels,
            ),
        }
    }
}

impl std::error::Error for VadError {}
