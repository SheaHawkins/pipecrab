//! The [`Transcriber`] trait and its [`SttError`].

use async_trait::async_trait;
use pipecrab_core::AudioFormat;
use pipecrab_runtime::MaybeSendSync;
use std::sync::Arc;

/// The swappable speech-to-text capability: `f32` samples in, a transcript out.
///
/// This is the durable interface. A native engine (`ort`) and a browser engine
/// (Transformers.js in a Web Worker) both implement this one trait, so
/// [`SttStage`](crate::SttStage) — and the pipeline above it — never names a
/// concrete model. The offload decision lives in the *impl* (native offloads to
/// a worker thread; wasm awaits a Web Worker), so the stage stays engine-neutral
/// and just `.await`s [`transcribe`](Transcriber::transcribe).
///
/// `?Send` matches pipecrab's single-threaded execution model, so one
/// implementation runs unchanged on a current-thread executor and on `wasm32`,
/// where `Send` bounds cannot be satisfied.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait Transcriber: MaybeSendSync {
    /// The one format this engine accepts. The stage caches it and enforces it
    /// *before* feeding — see the crate-level [format authority](crate) note.
    /// Sync and infallible: it is known at construction, so it is callable from
    /// a stage's `decide_*` under the control-call carve-out.
    fn input_format(&self) -> AudioFormat;

    /// Transcribe `samples` (interleaved `f32` PCM) to text. Shared ownership
    /// lets a worker-backed implementation retain or enqueue the buffer without
    /// copying its samples.
    ///
    /// Samples are interpreted as [`input_format()`](Self::input_format): an
    /// `Arc<[f32]>` carries no sample rate, so no runtime detection is possible.
    /// Feeding a mismatch is a wiring bug the stage rejects fatally before a
    /// sample reaches here, so this method never has to.
    ///
    /// Takes `&self`: like every
    /// [`Stage::perform`](pipecrab_runtime::Stage::perform), transcription must
    /// not mutate observable state, so the run loop can drop an in-flight call on
    /// a barge-in interrupt without tearing anything.
    async fn transcribe(&self, samples: Arc<[f32]>) -> Result<String, SttError>;
}

/// Why a [`Transcriber::transcribe`] call failed.
///
/// Mirrors the message-plus-kind shape of the pipeline's other error types so
/// the conversion at the stage boundary
/// (`impl From<SttError> for StageError`) is direct. There is deliberately no
/// format-mismatch variant: samples are interpreted as
/// [`input_format()`](Transcriber::input_format) and the stage enforces that
/// format fatally, so an engine never sees nonconforming audio to reject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SttError {
    /// The transcription engine itself failed — an inference error, a worker
    /// that crashed, a model that never loaded. Carries a human-readable
    /// description.
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
