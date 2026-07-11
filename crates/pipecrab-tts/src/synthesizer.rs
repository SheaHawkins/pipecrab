//! The [`Synthesizer`] trait, its [`TtsError`], and the [`TtsAudioStream`] alias.

use async_trait::async_trait;
use pipecrab_core::{AudioChunk, AudioFormat};
use pipecrab_runtime::MaybeSendSync;

/// The audio a [`Synthesizer::synthesize`] call yields: a boxed stream of
/// [`AudioChunk`] results, delivered incrementally.
///
/// `BoxStream` on native and `LocalBoxStream` on `wasm32`, behind one cfg'd
/// alias — the same `Send`-where-it-exists split as
/// [`MaybeSend`](pipecrab_runtime::MaybeSend): the pipeline is one logical task
/// that stays `Send` for a work-stealing executor natively, while on `wasm32`
/// (one thread, `!Send` JS handles) that bound must vanish.
#[cfg(not(target_arch = "wasm32"))]
pub type TtsAudioStream = futures::stream::BoxStream<'static, Result<AudioChunk, TtsError>>;
/// The audio a [`Synthesizer::synthesize`] call yields: a boxed stream of
/// [`AudioChunk`] results, delivered incrementally.
#[cfg(target_arch = "wasm32")]
pub type TtsAudioStream = futures::stream::LocalBoxStream<'static, Result<AudioChunk, TtsError>>;

/// The swappable text-to-speech capability: text in, audio out incrementally.
///
/// This is the durable interface. A native engine and a browser engine (in a Web
/// Worker) both implement this one trait, so [`TtsStage`](crate::TtsStage) — and
/// the pipeline above it — never names a concrete model.
///
/// # Streaming is a barge-in requirement
///
/// [`synthesize`](Synthesizer::synthesize) yields audio a chunk at a time rather
/// than one buffer at the end: every stream item is a preemption point the run
/// loop can drop an in-flight synthesis at, so a user barging in stops playback
/// within one chunk instead of after a whole utterance. Dropping the stream is
/// how the *stage* stops pulling; [`cancel`](Synthesizer::cancel) is how the
/// *engine* stops producing.
///
/// [`cancel`](Synthesizer::cancel) is a *control call* (see
/// [`Processor`](pipecrab_core::Processor)'s control-call carve-out): it flips an
/// atomic the engine's worker observes, so it is synchronous, non-blocking, and
/// safe to invoke directly from a stage's `decide_*` where the barge-in is
/// decided. [`synthesize`](Synthesizer::synthesize) is async because it hands
/// text to that worker and returns its audio stream.
///
/// `?Send` on `wasm32` matches pipecrab's single-threaded execution model, so
/// one implementation runs unchanged on a current-thread executor and in the
/// browser, where `Send` bounds cannot be satisfied.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait Synthesizer: MaybeSendSync {
    /// The [`AudioFormat`] of every chunk this engine yields (e.g. 24 kHz mono).
    ///
    /// Unlike an STT or VAD engine, a synthesizer does not *accept* an input
    /// format to reject — it *produces* audio, so it reports the one format its
    /// chunks arrive in. Resampling to the playback rate belongs to a separate
    /// stage.
    fn output_format(&self) -> AudioFormat;

    /// Synthesize `text`, yielding audio incrementally.
    ///
    /// Takes `&self`: like every [`Stage::perform`](pipecrab_runtime::Stage::perform),
    /// synthesis must not mutate observable state, so the run loop can drop an
    /// in-flight call — at any stream item — on a barge-in interrupt without
    /// tearing anything. Every item of the returned [`TtsAudioStream`] is such a
    /// preemption point.
    async fn synthesize(&self, text: &str) -> Result<TtsAudioStream, TtsError>;

    /// Control call: stop in-flight synthesis. Sync, non-blocking, idempotent.
    ///
    /// Flips a flag the engine's worker observes; the next
    /// [`synthesize`](Synthesizer::synthesize) starts clean. Safe to call from a
    /// stage's synchronous `decide_*` — see the trait-level note.
    fn cancel(&self);
}

/// Why a [`Synthesizer::synthesize`] call failed.
///
/// Mirrors the message-plus-kind shape of the pipeline's other error types (e.g.
/// `pipecrab-stt`'s `SttError`) so the conversion at the stage boundary
/// (`impl From<TtsError> for StageError`) is direct. A synthesizer produces
/// audio rather than consuming a caller-chosen format, so it has no
/// `UnsupportedFormat` variant to reject with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtsError {
    /// The synthesis engine itself failed — an inference error, a worker that
    /// crashed, a model that never loaded. Carries a human-readable description.
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
