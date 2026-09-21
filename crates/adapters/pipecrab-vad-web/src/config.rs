//! [`SileroConfig`]: which model to fetch and the shape it expects.

use pipecrab_core::AudioFormat;

/// Where [`SileroScorer`](crate::SileroScorer) fetches its model, and the window
/// it feeds.
///
/// The defaults are Silero's 16 kHz branch — the only one the ONNX exports in
/// circulation keep — so a model URL is normally the whole configuration.
/// Threshold and hangover are *not* here: they belong to
/// [`DebounceConfig`](pipecrab_vad::DebounceConfig), one tier up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SileroConfig {
    /// URL the browser fetches the `.onnx` model from. Relative URLs resolve
    /// against the page, so a file served next to it is just `"silero_vad.onnx"`.
    pub model_url: String,
    /// Sample rate the model is fed at, passed to the model's `sr` input.
    pub sample_rate: u32,
    /// Samples per scored window: 512 for Silero at 16 kHz.
    pub window: usize,
}

impl SileroConfig {
    /// Fetch a model from `model_url` and run Silero's 16 kHz branch: 512-sample
    /// windows at 16 kHz mono.
    pub fn new(model_url: impl Into<String>) -> Self {
        Self {
            model_url: model_url.into(),
            sample_rate: 16_000,
            window: 512,
        }
    }

    /// The format the scorer's samples are interpreted as: `sample_rate`, mono.
    pub fn format(&self) -> AudioFormat {
        AudioFormat::new(self.sample_rate, 1)
    }
}
