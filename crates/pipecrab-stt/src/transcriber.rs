//! The [`Transcriber`] seam and its [`SttError`].

use async_trait::async_trait;
use pipecrab_core::AudioFormat;

/// The swappable speech-to-text capability: `f32` samples in, a transcript out.
///
/// This is the durable seam. A native engine (`ort`) and a browser engine
/// (Transformers.js in a Web Worker) both implement this one trait, so
/// [`SttStage`](crate::SttStage) — and the pipeline above it — never names a
/// concrete model. The offload decision lives in the *impl* (native offloads to
/// a worker thread; wasm awaits a Web Worker), so the stage stays engine-neutral
/// and just `.await`s [`transcribe`](Transcriber::transcribe).
///
/// `?Send` matches pipecrab's single-threaded execution model, so one
/// implementation runs unchanged on a current-thread executor and on `wasm32`,
/// where `Send` bounds cannot be satisfied.
#[async_trait(?Send)]
pub trait Transcriber {
    /// Transcribe `samples` (interleaved `f32` PCM in `format`) to text.
    ///
    /// Takes `&self`: like every
    /// [`Stage::perform`](pipecrab_runtime::Stage::perform), transcription must
    /// not mutate observable state, so the run loop can drop an in-flight call on
    /// a barge-in interrupt without tearing anything.
    ///
    /// An impl that accepts only one format (Moonshine wants 16 kHz mono) should
    /// reject a mismatch with [`SttError::UnsupportedFormat`] rather than
    /// resample — resampling belongs to a separate stage.
    async fn transcribe(&self, samples: &[f32], format: AudioFormat) -> Result<String, SttError>;
}

/// Why a [`Transcriber::transcribe`] call failed.
///
/// Mirrors the message-plus-kind shape of the pipeline's other error types so
/// the conversion at the stage boundary
/// (`impl From<SttError> for StageError`) is direct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SttError {
    /// The transcription engine itself failed — an inference error, a worker
    /// that crashed, a model that never loaded. Carries a human-readable
    /// description.
    Engine(String),
    /// The transcriber requires a specific input format and got another. It does
    /// not resample: feed it audio in the format it expects.
    UnsupportedFormat {
        /// The format this transcriber accepts (e.g. 16 kHz mono).
        expected: AudioFormat,
        /// The format of the rejected samples.
        got: AudioFormat,
    },
}

impl std::fmt::Display for SttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SttError::Engine(msg) => write!(f, "stt engine error: {msg}"),
            SttError::UnsupportedFormat { expected, got } => write!(
                f,
                "stt format mismatch: transcriber expects {} Hz/{} ch, got {} Hz/{} ch",
                expected.sample_rate, expected.channels, got.sample_rate, got.channels,
            ),
        }
    }
}

impl std::error::Error for SttError {}
