//! [`WebAudioConfig`]: chunk size and queue depth for [`WebAudioSource`].
//!
//! [`WebAudioSource`]: crate::WebAudioSource

/// How [`WebAudioSource`](crate::WebAudioSource) chunks and buffers capture.
///
/// There is no device selection: the browser picks the input device (and offers
/// the user the choice in its own permission UI), so a `DeviceSelection`
/// counterpart to the cpal config would have nothing to select.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebAudioConfig {
    /// Milliseconds of audio per emitted chunk. The worklet rounds this to whole
    /// frames at the context's rate, and never below one 128-frame render quantum.
    pub chunk_ms: u32,
    /// Chunks the queue holds before the worklet's next chunk is dropped as an
    /// overrun. Depth in time is `chunk_ms * queue_chunks`.
    pub queue_chunks: usize,
}

impl Default for WebAudioConfig {
    /// 20 ms chunks, 32 of them queued (~640 ms of slack).
    fn default() -> Self {
        Self {
            chunk_ms: 20,
            queue_chunks: 32,
        }
    }
}
