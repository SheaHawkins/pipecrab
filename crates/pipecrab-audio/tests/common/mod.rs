//! Hardware-free [`AudioSource`] / [`AudioSink`] fixtures shared across this
//! crate's integration tests. Kept out of the shipped `src`: these are test
//! scaffolding, not public API.
//!
//! [`MockSource`] replays a fixed script of chunks — build one straight from a
//! ramp with [`MockSource::ramp`]. [`MockSink`] records everything it is asked
//! to play so a test can assert on the samples that came out. No devices, no
//! threads.
#![allow(dead_code)] // each test binary uses only a subset of these helpers.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use pipecrab_audio::{AudioChunk, AudioError, AudioFormat, AudioSink, AudioSource};

/// An [`AudioSource`] that yields a predetermined list of chunks, then `None`.
pub struct MockSource {
    format: AudioFormat,
    queue: VecDeque<AudioChunk>,
}

impl MockSource {
    /// A source that yields each of `chunks` in order (restamped with `format`),
    /// then `None`.
    pub fn new(format: AudioFormat, chunks: impl IntoIterator<Item = Arc<[f32]>>) -> Self {
        let queue = chunks.into_iter().map(|samples| AudioChunk::new(samples, format)).collect();
        Self { format, queue }
    }

    /// A source whose entire output is the ramp `0.0, 1.0, 2.0, …` — one value
    /// per sample — split into `chunks` chunks of `chunk_frames` samples each.
    ///
    /// The values are exact and monotonic, so a passthrough test can flatten the
    /// sink's output and compare it to the same ramp.
    pub fn ramp(format: AudioFormat, chunk_frames: usize, chunks: usize) -> Self {
        let mut queue = VecDeque::with_capacity(chunks);
        for c in 0..chunks {
            let start = (c * chunk_frames) as u32;
            let samples: Arc<[f32]> = (0..chunk_frames).map(|i| (start + i as u32) as f32).collect();
            queue.push_back(AudioChunk::new(samples, format));
        }
        Self { format, queue }
    }
}

#[async_trait(?Send)]
impl AudioSource for MockSource {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn next_chunk(&mut self) -> Option<AudioChunk> {
        self.queue.pop_front()
    }
}

/// An [`AudioSink`] that records every chunk it is asked to play.
pub struct MockSink {
    format: AudioFormat,
    received: Vec<AudioChunk>,
}

impl MockSink {
    /// A sink expecting chunks in `format`, with nothing recorded yet.
    pub fn new(format: AudioFormat) -> Self {
        Self { format, received: Vec::new() }
    }

    /// The chunks played so far, in arrival order.
    pub fn chunks(&self) -> &[AudioChunk] {
        &self.received
    }

    /// Every received sample, flattened into one buffer in arrival order.
    pub fn samples(&self) -> Vec<f32> {
        self.received.iter().flat_map(|c| c.samples.iter().copied()).collect()
    }
}

#[async_trait(?Send)]
impl AudioSink for MockSink {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn play(&mut self, chunk: AudioChunk) -> Result<(), AudioError> {
        self.received.push(chunk);
        Ok(())
    }
}
