//! Streaming transcription: the [`StreamingTranscriber`] protocol for engines
//! that emit partial hypotheses while audio is still arriving, plus the
//! [`Buffered`] adapter that fits a one-shot [`Transcriber`] to that protocol.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pipecrab_core::AudioFormat;
use pipecrab_runtime::MaybeSendSync;

use crate::{SttError, Transcriber};

/// A transcription session protocol for engines that emit partial results
/// while audio is still arriving (e.g. a streaming Zipformer). One active
/// utterance per engine instance.
///
/// # Engines are worker-handles
///
/// An implementor is expected to be a thin *handle* to a long-lived worker that
/// owns the mutable decoder state: a dedicated thread on native, a Web Worker on
/// `wasm32`. That is why every method takes `&self` — they are cheap
/// message-passes to the worker, not the inference itself — and why the worker
/// outlives any single call. A barge-in that drops an in-flight
/// [`feed`](Self::feed) future does **not** reset the worker; only
/// [`cancel`](Self::cancel) does.
///
/// [`cancel`](Self::cancel) is a *control call* (see
/// [`Processor`](pipecrab_core::Processor)'s control-call carve-out): it flips
/// an atomic the worker observes on its next step, so it is synchronous,
/// non-blocking, and safe to invoke from a stage's `decide_*`. The other three
/// methods are async because they exchange messages with the worker.
///
/// `?Send` on `wasm32` matches pipecrab's single-threaded execution model, so
/// one implementation runs unchanged on a current-thread executor and in the
/// browser, where `Send` bounds cannot be satisfied.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait StreamingTranscriber: MaybeSendSync {
    /// The one format this engine accepts. The stage caches it and enforces it
    /// *before* feeding — see the crate-level [format authority](crate) note.
    /// Sync and infallible: known at construction, so callable from a stage's
    /// `decide_*` under the control-call carve-out.
    fn input_format(&self) -> AudioFormat;

    /// Open an utterance. Only one is active per engine instance, so the caller
    /// must close the previous one — via [`end_utterance`](Self::end_utterance)
    /// or [`cancel`](Self::cancel) — before opening another; calling this while
    /// an utterance is already active is a protocol violation and an engine
    /// rejects it rather than silently discarding the in-progress utterance.
    ///
    /// No format parameter: samples fed to this utterance are interpreted as
    /// [`input_format()`](Self::input_format), which the stage has already
    /// enforced.
    async fn begin_utterance(&self) -> Result<(), SttError>;

    /// Feed one window of samples; returns whatever events are ready so far.
    /// Cheap message-pass to the engine's worker.
    async fn feed(&self, samples: &[f32]) -> Result<Vec<SttEvent>, SttError>;

    /// Close the utterance; drains remaining events, including the
    /// [`Final`](SttEvent::Final).
    async fn end_utterance(&self) -> Result<Vec<SttEvent>, SttError>;

    /// Control call: stop in-flight work and discard the active utterance.
    /// Sync, non-blocking, idempotent. The next
    /// [`begin_utterance`](Self::begin_utterance) starts clean.
    fn cancel(&self);
}

/// An event emitted by a [`StreamingTranscriber`] as an utterance progresses.
#[derive(Clone, Debug, PartialEq)]
pub enum SttEvent {
    /// In-progress hypothesis. `stable` follows the core
    /// [`Transcript`](pipecrab_core::Transcript) invariant: `text[..stable]` is
    /// frozen and only the tail beyond it may still change.
    Partial {
        /// The current best-guess transcript for the utterance so far.
        text: Arc<str>,
        /// Byte length of the frozen prefix; on a char boundary and
        /// `<= text.len()`.
        stable: usize,
    },
    /// The utterance's completed transcript.
    Final(Arc<str>),
    /// The engine's own end-of-utterance signal, if it does internal
    /// endpointing. v1: the stage logs and otherwise ignores this (a future
    /// `TurnEnded` frame is out of scope).
    Endpoint,
}

/// Fits a one-shot [`Transcriber`] to the [`StreamingTranscriber`] protocol by
/// buffering the whole utterance and transcribing it once at the end — the
/// adapter for chunk-final engines like Moonshine that have no partial output.
///
/// It emits **no** partials: [`feed`](StreamingTranscriber::feed) only
/// accumulates and returns `[]`, and the entire transcript arrives as a single
/// [`SttEvent::Final`] from [`end_utterance`](StreamingTranscriber::end_utterance).
///
/// # One honest limitation
///
/// **Cancel cannot stop inference mid-flight.**
/// [`cancel`](StreamingTranscriber::cancel) clears the buffer and marks any
/// in-flight transcription stale so its result is discarded when it returns,
/// but the underlying one-shot inference — offloaded and detached — still runs
/// to completion off-thread. True mid-inference cancel requires a native
/// streaming engine.
pub struct Buffered<T: Transcriber> {
    inner: T,
    state: Mutex<BufferedState>,
}

/// The mutable session state, behind a [`Mutex`] because the trait methods take
/// `&self`. The lock is never held across an `.await`, so it stays uncontended
/// and `cancel`'s critical section never blocks.
struct BufferedState {
    /// Whether an utterance is open — set by `begin_utterance`, cleared by
    /// `end_utterance`/`cancel`. It only gates the protocol; the format itself
    /// is no longer tracked (the stage enforces it before a sample arrives).
    active: bool,
    /// Accumulated interleaved samples for the active utterance.
    buffer: Vec<f32>,
    /// Bumped by `cancel`. `end_utterance` snapshots it before awaiting and
    /// discards its result if the value changed while it was in flight.
    generation: u64,
}

impl<T: Transcriber> Buffered<T> {
    /// Wrap a one-shot `transcriber` as a chunk-final streaming engine.
    pub fn new(transcriber: T) -> Self {
        Self {
            inner: transcriber,
            state: Mutex::new(BufferedState { active: false, buffer: Vec::new(), generation: 0 }),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<T: Transcriber> StreamingTranscriber for Buffered<T> {
    fn input_format(&self) -> AudioFormat {
        self.inner.input_format()
    }

    async fn begin_utterance(&self) -> Result<(), SttError> {
        let mut st = self.state.lock().expect("Buffered state mutex poisoned");
        // One active utterance per instance: refuse a second begin rather than
        // silently dropping the in-progress one. The caller must `end_utterance`
        // or `cancel` to close it first.
        if st.active {
            return Err(SttError::Engine(
                "Buffered::begin_utterance called while an utterance is already active".into(),
            ));
        }
        st.active = true;
        st.buffer.clear();
        Ok(())
    }

    async fn feed(&self, samples: &[f32]) -> Result<Vec<SttEvent>, SttError> {
        let mut st = self.state.lock().expect("Buffered state mutex poisoned");
        if !st.active {
            return Err(SttError::Engine(
                "Buffered::feed called without an active utterance".into(),
            ));
        }
        st.buffer.extend_from_slice(samples);
        // Chunk-final: nothing to report until the utterance closes.
        Ok(Vec::new())
    }

    async fn end_utterance(&self) -> Result<Vec<SttEvent>, SttError> {
        // Snapshot the utterance under the lock, then release it before the
        // awaited inference — the guard must not cross the `.await`.
        let (samples, generation) = {
            let mut st = self.state.lock().expect("Buffered state mutex poisoned");
            if !st.active {
                return Err(SttError::Engine(
                    "Buffered::end_utterance called without a begin_utterance".into(),
                ));
            }
            st.active = false;
            (std::mem::take(&mut st.buffer), st.generation)
        };

        // One-shot inference over the whole utterance. The wrapped engine owns
        // where this runs; if a barge-in drops this future, that offloaded work
        // detaches and its result is lost.
        let text = self.inner.transcribe(&samples).await?;

        // If `cancel` bumped the generation while we were awaiting, the utterance
        // was abandoned: discard the now-stale transcript.
        let stale = {
            let st = self.state.lock().expect("Buffered state mutex poisoned");
            st.generation != generation
        };
        if stale {
            return Ok(Vec::new());
        }
        Ok(vec![SttEvent::Final(text.into())])
    }

    fn cancel(&self) {
        // Uncontended (the lock never crosses an await), so this control call is
        // effectively non-blocking. Clear the pending audio and bump the
        // generation so any in-flight `end_utterance` discards its result.
        let mut st = self.state.lock().expect("Buffered state mutex poisoned");
        st.active = false;
        st.buffer.clear();
        st.generation = st.generation.wrapping_add(1);
    }
}
