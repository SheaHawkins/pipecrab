//! pipecrab-audio: the platform-neutral boundary where audio enters and leaves a
//! pipeline.
//!
//! [`pipecrab-core`](pipecrab_core) owns the *frame* — [`AudioChunk`] and
//! [`AudioFormat`], re-exported here for convenience. This crate owns *how*
//! audio crosses the pipeline edge: the [`AudioSource`] and [`AudioSink`]
//! traits. Concrete backends (e.g. cpal for desktop) live in their own crates
//! behind these same traits; hardware-free mock implementations live alongside
//! this crate's tests.
//!
//! Audio is a first-party pipeline payload — a typed
//! [`DataFrame::Audio`](pipecrab_core::DataFrame::Audio), never a `Custom`
//! frame — so nothing here downcasts.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use async_trait::async_trait;

pub use pipecrab_core::{AudioChunk, AudioFormat};

/// Why an [`AudioSink::play`] (or an underlying backend I/O) failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioError {
    /// The sink starved — no audio was ready when the device needed it.
    Underrun,
    /// The device or stream is closed; no more audio can flow.
    Closed,
    /// A device- or backend-level failure, with a human-readable description.
    Device(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::Underrun => write!(f, "audio underrun"),
            AudioError::Closed => write!(f, "audio device closed"),
            AudioError::Device(msg) => write!(f, "audio device error: {msg}"),
        }
    }
}

impl std::error::Error for AudioError {}

/// A source of audio flowing *into* a pipeline (device capture, file, network,
/// or a mock).
///
/// `?Send` matches pipecrab's single-threaded execution model, so one
/// implementation runs unchanged on a current-thread executor and on `wasm32`.
#[async_trait(?Send)]
pub trait AudioSource {
    /// The format of the chunks this source yields. Fixed for the source's life.
    fn format(&self) -> AudioFormat;

    /// Await the next chunk, or `None` once the source is exhausted or closed.
    async fn next_chunk(&mut self) -> Option<AudioChunk>;
}

/// A sink of audio flowing *out of* a pipeline (device playback, file, network,
/// or a mock).
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
