//! One-shot speech transcription.

use async_trait::async_trait;
use pipecrab_core::AudioFormat;
use pipecrab_runtime::MaybeSendSync;

/// Transcribes `f32` PCM samples to text.
///
/// Implementations own their worker or offloading strategy.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait Transcriber: MaybeSendSync {
    /// The format this engine accepts.
    fn input_format(&self) -> AudioFormat;

    /// Transcribe `samples` (interleaved `f32` PCM) to text.
    ///
    /// Samples use [`Self::input_format`]. The stage rejects mismatches before
    /// calling this method.
    ///
    /// The returned future must be safe to drop at an await point.
    async fn transcribe(&self, samples: &[f32]) -> Result<String, SttError>;
}

/// An error from [`Transcriber::transcribe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SttError {
    /// A transcription engine failure.
    Engine(String),
}

impl std::fmt::Display for SttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SttError::Engine(msg) => write!(f, "stt engine error: {msg}"),
        }
    }
}

impl std::error::Error for SttError {}
