//! Streaming text-to-speech.

use async_trait::async_trait;
use pipecrab_core::{AudioChunk, AudioFormat};
use pipecrab_runtime::MaybeSendSync;

/// A stream of synthesized [`AudioChunk`]s.
#[cfg(not(target_arch = "wasm32"))]
pub type TtsAudioStream = futures::stream::BoxStream<'static, Result<AudioChunk, TtsError>>;
/// A stream of synthesized [`AudioChunk`]s.
#[cfg(target_arch = "wasm32")]
pub type TtsAudioStream = futures::stream::LocalBoxStream<'static, Result<AudioChunk, TtsError>>;

/// Synthesizes text into audio incrementally.
///
/// Implementations should be handles to workers that own mutable engine state;
/// methods use `&self` so stage effects remain safe to cancel.
///
/// # Streaming is a barge-in requirement
///
/// Each [`TtsAudioStream`] item is a preemption point. Dropping the stream stops
/// consumption; [`Synthesizer::cancel`] stops engine production.
///
/// [`Synthesizer::cancel`] is a [`pipecrab_core::Processor`] control call: it
/// must be synchronous, non-blocking, and idempotent.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait Synthesizer: MaybeSendSync {
    /// The [`AudioFormat`] of every chunk this engine yields.
    ///
    /// Resampling to the playback format belongs in another stage.
    fn output_format(&self) -> AudioFormat;

    /// Synthesize `text`, yielding audio incrementally.
    ///
    /// The returned stream must be safe to drop between items.
    async fn synthesize(&self, text: &str) -> Result<TtsAudioStream, TtsError>;

    /// Stops in-flight synthesis synchronously and idempotently.
    fn cancel(&self);
}

/// An error from [`Synthesizer::synthesize`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtsError {
    /// A synthesis engine failure.
    Engine(String),
}

impl std::fmt::Display for TtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtsError::Engine(msg) => write!(f, "tts engine error: {msg}"),
        }
    }
}

impl std::error::Error for TtsError {}
