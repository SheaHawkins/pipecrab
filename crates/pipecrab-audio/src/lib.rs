//! pipecrab-audio: the platform-neutral boundary where audio enters and leaves a
//! pipeline.
//!
//! [`pipecrab-core`](pipecrab_core) owns the *frame* — [`AudioChunk`] and
//! [`AudioFormat`], re-exported here for convenience. This crate owns *how*
//! audio crosses the pipeline edge: the [`AudioSource`] and [`AudioSink`] traits
//! plus hardware-free [`mock`] implementations for tests. Concrete backends
//! (e.g. cpal for desktop) live in their own crates behind these same traits.
//!
//! Audio is a first-party pipeline payload — a typed
//! [`DataFrame::Audio`](pipecrab_core::DataFrame::Audio), never a `Custom`
//! frame — so nothing here downcasts.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use async_trait::async_trait;

pub mod mock;

pub use pipecrab_core::{AudioChunk, AudioFormat};

/// Why an [`AudioSource`]/[`AudioSink`] (or an underlying backend I/O) failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioError {
    /// The sink starved — no audio was ready when the device needed it.
    Underrun,
    /// The device or stream is closed; no more audio can flow.
    Closed,
    /// A chunk's format did not match the sink's format. The sink does not
    /// resample: the caller must feed it chunks in the sink's own format.
    FormatMismatch {
        /// The format the sink accepts (its own [`AudioSink::format`]).
        expected: AudioFormat,
        /// The format of the rejected chunk.
        got: AudioFormat,
    },
    /// A device- or backend-level failure, with a human-readable description.
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

/// A source of audio flowing *into* a pipeline (device capture, file, network,
/// or a [`mock`]).
///
/// `?Send` matches pipecrab's single-threaded execution model, so one
/// implementation runs unchanged on a current-thread executor and on `wasm32`.
#[async_trait(?Send)]
pub trait AudioSource {
    /// The format of the chunks this source yields. Fixed for the source's life.
    fn format(&self) -> AudioFormat;

    /// Await the next chunk.
    ///
    /// Three outcomes, kept distinct so a live device can report failure without
    /// being mistaken for a clean end of stream:
    /// - `Ok(Some(chunk))` — a chunk of audio.
    /// - `Ok(None)` — the source is *gracefully* exhausted (e.g. a file or mock
    ///   ran out); there will be no more chunks.
    /// - `Err(_)` — the source *failed* (e.g. the capture device dropped out).
    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>, AudioError>;
}

/// A sink of audio flowing *out of* a pipeline (device playback, file, network,
/// or a [`mock`]).
///
/// `?Send` for the same reason as [`AudioSource`].
#[async_trait(?Send)]
pub trait AudioSink {
    /// The format this sink expects incoming chunks to be in.
    fn format(&self) -> AudioFormat;

    /// Play (or enqueue) one chunk. May `.await` to apply backpressure when the
    /// sink is full; returns an [`AudioError`] if the chunk cannot be accepted.
    async fn play(&mut self, chunk: AudioChunk) -> Result<(), AudioError>;
}
