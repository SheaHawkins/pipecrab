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

use pipecrab_runtime::{maybe_async_trait, MaybeSend};

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

// Doc comments live *inside* `maybe_async_trait!` so they attach to the trait
// (an outer attribute placed before the macro call would document the call, not
// the item, and be dropped).
maybe_async_trait! {
    /// A source of audio flowing *into* a pipeline (device capture, file,
    /// network, or a [`mock`]).
    ///
    /// Carries [`MaybeSend`] like every capability interface: `Send` on native so a
    /// per-session pump task is spawnable on a multi-threaded executor (a WebRTC
    /// server runs one source per call), vacuous on `wasm32` where `Send` cannot
    /// be satisfied. It is `MaybeSend`, not `MaybeSendSync`: [`next_chunk`] takes
    /// `&mut self`, so the future needs to *move* `self`, never share it. A
    /// backend whose native handle is `!Send` (cpal's `Stream`) keeps that handle
    /// off the struct — behind a stream-owning thread — rather than leaking
    /// `?Send` here.
    ///
    /// [`next_chunk`]: AudioSource::next_chunk
    pub trait AudioSource: MaybeSend {
        /// The format of the chunks this source yields. Fixed for the source's life.
        fn format(&self) -> AudioFormat;

        /// Await the next chunk.
        ///
        /// Three outcomes, kept distinct so a live device can report failure
        /// without being mistaken for a clean end of stream:
        /// - `Ok(Some(chunk))` — a chunk of audio.
        /// - `Ok(None)` — the source is *gracefully* exhausted (e.g. a file or
        ///   mock ran out); there will be no more chunks.
        /// - `Err(_)` — the source *failed* (e.g. the capture device dropped out).
        async fn next_chunk(&mut self) -> Result<Option<AudioChunk>, AudioError>;
    }
}

maybe_async_trait! {
    /// A sink of audio flowing *out of* a pipeline (device playback, file,
    /// network, or a [`mock`]).
    ///
    /// [`MaybeSend`] for the same reason as [`AudioSource`]: `play` takes
    /// `&mut self`, so `Send` (native) / vacuous (`wasm32`) is the exact bound.
    pub trait AudioSink: MaybeSend {
        /// The format this sink expects incoming chunks to be in.
        fn format(&self) -> AudioFormat;

        /// Play (or enqueue) one chunk. May `.await` to apply backpressure when
        /// the sink is full; returns an [`AudioError`] if the chunk cannot be
        /// accepted.
        async fn play(&mut self, chunk: AudioChunk) -> Result<(), AudioError>;
    }
}
