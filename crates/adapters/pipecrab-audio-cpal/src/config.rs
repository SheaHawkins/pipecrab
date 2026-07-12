//! Shared device and buffering configuration for [`CpalSource`](crate::CpalSource)
//! and [`CpalSink`](crate::CpalSink).

/// Which device to open.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum DeviceSelection {
    /// The host's default device for this direction.
    #[default]
    Default,
    /// The device whose name matches this string — one of
    /// [`input_device_names`](crate::input_device_names) /
    /// [`output_device_names`](crate::output_device_names).
    Name(String),
}

/// Configuration shared by [`crate::CpalSource`] and [`crate::CpalSink`].
///
/// The default selects both default devices with 20 ms chunks.
#[derive(Debug, Clone)]
pub struct CpalConfig {
    /// Input device the source captures from.
    pub source_device: DeviceSelection,
    /// Output device the sink plays to.
    pub sink_device: DeviceSelection,
    /// Target chunk duration in milliseconds; chunk frames = `rate * chunk_ms / 1000`.
    pub chunk_ms: u32,
    /// Ring capacity in chunks; larger values absorb jitter but add latency.
    pub ring_chunks: usize,
}

impl Default for CpalConfig {
    fn default() -> Self {
        Self {
            source_device: DeviceSelection::Default,
            sink_device: DeviceSelection::Default,
            chunk_ms: 20,
            ring_chunks: 8,
        }
    }
}

impl CpalConfig {
    /// Chunk size in frames at `sample_rate`.
    pub(crate) fn chunk_frames(&self, sample_rate: u32) -> usize {
        (sample_rate * self.chunk_ms / 1000) as usize
    }

    /// Ring capacity in samples at `sample_rate`.
    pub(crate) fn ring_capacity(&self, sample_rate: u32) -> usize {
        self.chunk_frames(sample_rate) * self.ring_chunks
    }
}
