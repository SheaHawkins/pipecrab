//! [`TransformersSttConfig`]: which model to load and how to decode with it.

use js_sys::{Object, Reflect};
use pipecrab_core::AudioFormat;
use wasm_bindgen::prelude::*;

/// How [`TransformersStt`](crate::TransformersStt) loads and drives a
/// Transformers.js pipeline.
///
/// Every field but `model` has a working default, so a Hub id is normally the
/// whole configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformersSttConfig {
    /// Hub id of the model to fetch, e.g. `"Xenova/whisper-tiny.en"`.
    pub model: String,
    /// Sample rate the model is fed at. Every ASR export in circulation wants
    /// 16 kHz mono.
    pub sample_rate: u32,
    /// Weight quantization: `"q8"`, `"fp16"`, `"fp32"`, … `None` leaves the
    /// engine's own default, which picks per device.
    pub dtype: Option<String>,
    /// Backend to run on: `"wasm"` or `"webgpu"`. `None` leaves the engine's
    /// default.
    pub device: Option<String>,
    /// Language to decode as, for a multilingual model. `None` lets the model
    /// detect one; English-only models reject the option entirely.
    pub language: Option<String>,
    /// Seconds of audio per decode window. An utterance longer than this is
    /// decoded in overlapping chunks and stitched, which is what keeps a
    /// two-minute utterance from overrunning a 30-second Whisper context.
    pub chunk_length_s: Option<f32>,
}

impl TransformersSttConfig {
    /// Load `model` from the Hub and decode 16 kHz mono in 30-second windows,
    /// leaving quantization, device, and language to the engine.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            sample_rate: 16_000,
            dtype: None,
            device: None,
            language: None,
            chunk_length_s: Some(30.0),
        }
    }

    /// The format the transcriber's samples are interpreted as: `sample_rate`, mono.
    pub fn format(&self) -> AudioFormat {
        AudioFormat::new(self.sample_rate, 1)
    }

    /// Options for `pipeline(...)`: how the model is loaded.
    pub(crate) fn pipeline_options(&self) -> Object {
        let options = Object::new();
        set(&options, "dtype", self.dtype.as_deref());
        set(&options, "device", self.device.as_deref());
        options
    }

    /// Options for each call: how the model decodes.
    pub(crate) fn decode_options(&self) -> Object {
        let options = Object::new();
        set(&options, "language", self.language.as_deref());
        if let Some(seconds) = self.chunk_length_s {
            let _ = Reflect::set(
                &options,
                &JsValue::from_str("chunk_length_s"),
                &JsValue::from_f64(f64::from(seconds)),
            );
        }
        options
    }
}

/// Set `key` on a fresh object when the value is present. Fallible only for
/// exotic receivers (a proxy, a frozen object); a plain `Object` never rejects.
fn set(options: &Object, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        let _ = Reflect::set(options, &JsValue::from_str(key), &JsValue::from_str(value));
    }
}
