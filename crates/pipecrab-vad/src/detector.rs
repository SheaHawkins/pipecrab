//! The [`VoiceActivityDetector`] seam and its [`VadError`].

use async_trait::async_trait;
use pipecrab_core::AudioFormat;
use pipecrab_runtime::MaybeSendSync;

/// The swappable voice-activity-detection capability: `f32` samples in, a
/// speech-probability out.
///
/// This is the durable seam. A native engine (Silero on `ort`) and a browser
/// engine both implement this one trait, so a downstream VAD stage — and the
/// pipeline above it — never names a concrete model. The offload decision lives
/// in the *impl* (native offloads to a worker thread; wasm awaits a Web Worker),
/// so the caller stays engine-neutral and just `.await`s
/// [`detect`](VoiceActivityDetector::detect).
///
/// `?Send` matches pipecrab's single-threaded execution model, so one
/// implementation runs unchanged on a current-thread executor and on `wasm32`,
/// where `Send` bounds cannot be satisfied.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait VoiceActivityDetector: MaybeSendSync {
    /// Score `samples` (interleaved `f32` PCM in `format`) and return the
    /// probability in `0.0..=1.0` that the window contains speech.
    ///
    /// Takes `&self`: like every
    /// [`Stage::perform`](pipecrab_runtime::Stage::perform), detection must not
    /// mutate observable state, so the run loop can drop an in-flight call on a
    /// barge-in interrupt without tearing anything.
    ///
    /// An impl that accepts only one format (Silero wants 16 kHz mono) should
    /// reject a mismatch with [`VadError::UnsupportedFormat`] rather than
    /// resample — resampling belongs to a separate stage.
    async fn detect(&self, samples: &[f32], format: AudioFormat) -> Result<f32, VadError>;
}

/// Why a [`VoiceActivityDetector::detect`] call failed.
///
/// Mirrors the message-plus-kind shape of pipecrab's other seam error types so
/// the conversion at a stage boundary is direct.
#[derive(Debug, Clone, PartialEq)]
pub enum VadError {
    /// The detection engine itself failed — an inference error, a worker that
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
