//! Streaming transcription and the [`Buffered`] one-shot adapter.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pipecrab_core::AudioFormat;
use pipecrab_runtime::MaybeSendSync;

use crate::{SttError, Transcriber};

/// A streaming transcription session with one active utterance.
///
/// # Engines are worker-handles
///
/// Implementations should be handles to workers that own decoder state. Dropping
/// an in-flight [`StreamingTranscriber::feed`] does not reset the worker;
/// [`StreamingTranscriber::cancel`] does.
///
/// [`StreamingTranscriber::cancel`] is a [`pipecrab_core::Processor`] control call.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait StreamingTranscriber: MaybeSendSync {
    /// The format this engine accepts.
    fn input_format(&self) -> AudioFormat;

    /// Opens an utterance.
    ///
    /// A second call before [`Self::end_utterance`] or [`Self::cancel`] is a
    /// protocol violation.
    async fn begin_utterance(&self) -> Result<(), SttError>;

    /// Feeds one sample window and returns available events.
    async fn feed(&self, samples: &[f32]) -> Result<Vec<SttEvent>, SttError>;

    /// Closes the utterance and drains remaining events.
    async fn end_utterance(&self) -> Result<Vec<SttEvent>, SttError>;

    /// Stops in-flight work and discards the active utterance.
    fn cancel(&self);
}

/// An event emitted by a [`StreamingTranscriber`] as an utterance progresses.
#[derive(Clone, Debug, PartialEq)]
pub enum SttEvent {
    /// In-progress hypothesis following the [`pipecrab_core::Transcript`] invariant.
    Partial {
        /// The current hypothesis.
        text: Arc<str>,
        /// Byte length of the fixed prefix.
        stable: usize,
    },
    /// The utterance's completed transcript.
    Final(Arc<str>),
    /// An engine-detected utterance endpoint.
    Endpoint,
}

/// Adapts a one-shot [`Transcriber`] to [`StreamingTranscriber`].
///
/// [`StreamingTranscriber::feed`] buffers samples without emitting partials.
/// [`StreamingTranscriber::end_utterance`] emits one [`SttEvent::Final`].
///
/// # Cancellation
///
/// [`StreamingTranscriber::cancel`] discards an in-flight result but cannot stop
/// the underlying one-shot inference.
pub struct Buffered<T: Transcriber> {
    inner: T,
    state: Mutex<BufferedState>,
}

/// Mutable session state. The lock is never held across an await point.
struct BufferedState {
    /// Whether an utterance is open.
    active: bool,
    /// Accumulated interleaved samples for the active utterance.
    buffer: Vec<f32>,
    /// Generation counter for discarding stale results after cancellation.
    generation: u64,
}

impl<T: Transcriber> Buffered<T> {
    /// Wrap a one-shot `transcriber` as a chunk-final streaming engine.
    pub fn new(transcriber: T) -> Self {
        Self {
            inner: transcriber,
            state: Mutex::new(BufferedState {
                active: false,
                buffer: Vec::new(),
                generation: 0,
            }),
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
