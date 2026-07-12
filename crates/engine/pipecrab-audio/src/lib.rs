//! Platform-neutral audio sources and sinks.
//!
//! [`AudioSource`] and [`AudioSink`] define how audio crosses pipeline
//! boundaries. Backends implement these traits without exposing device types to
//! the pipeline. [`AudioChunk`] and [`AudioFormat`] are re-exported from core.
//!
//! Audio travels as [`pipecrab_core::DataFrame::Audio`].
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use pipecrab_runtime::{maybe_async_trait, MaybeSend};

pub mod mock;

pub use pipecrab_core::{AudioChunk, AudioFormat};

/// An audio source, sink, or backend error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioError {
    /// No audio was ready when the sink needed it.
    Underrun,
    /// The device or stream is closed.
    Closed,
    /// A chunk's format did not match the sink's format. The sink does not
    /// resample: the caller must feed it chunks in the sink's own format.
    FormatMismatch {
        /// The format the sink accepts (its own [`AudioSink::format`]).
        expected: AudioFormat,
        /// The format of the rejected chunk.
        got: AudioFormat,
    },
    /// A device or backend failure.
    Device(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::Underrun => write!(f, "audio underrun"),
            AudioError::Closed => write!(f, "audio device closed"),
            AudioError::FormatMismatch { expected, got } => write!(
                f,
                "audio format mismatch: sink expects {} Hz/{} ch, got {} Hz/{} ch",
                expected.sample_rate, expected.channels, got.sample_rate, got.channels,
            ),
            AudioError::Device(msg) => write!(f, "audio device error: {msg}"),
        }
    }
}

impl std::error::Error for AudioError {}

// Doc comments live *inside* `maybe_async_trait!` so they attach to the trait
// (an outer attribute placed before the macro call would document the call, not
// the item, and be dropped).
maybe_async_trait! {
    /// A source of audio flowing *into* a pipeline (device capture, file,
    /// network, or a [`mock`]).
    ///
    /// [`MaybeSend`] permits native executors to move the source while remaining
    /// vacuous on `wasm32`. The mutable method means sources need not be `Sync`.
    pub trait AudioSource: MaybeSend {
        /// The format of the chunks this source yields. Fixed for the source's life.
        fn format(&self) -> AudioFormat;

        /// Await the next chunk.
        ///
        /// The result distinguishes data, graceful exhaustion, and failure:
        /// - `Ok(Some(chunk))` — a chunk of audio.
        /// - `Ok(None)` — the source is exhausted.
        /// - `Err(_)` — the source failed.
        async fn next_chunk(&mut self) -> Result<Option<AudioChunk>, AudioError>;
    }
}

maybe_async_trait! {
    /// A sink of audio flowing *out of* a pipeline (device playback, file,
    /// network, or a [`mock`]).
    ///
    /// Like [`AudioSource`], it is movable on native targets but need not be `Sync`.
    pub trait AudioSink: MaybeSend {
        /// The format this sink expects incoming chunks to be in.
        fn format(&self) -> AudioFormat;

        /// Plays or enqueues a chunk, awaiting backpressure when full.
        async fn play(&mut self, chunk: AudioChunk) -> Result<(), AudioError>;
    }
}
