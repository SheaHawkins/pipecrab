//! [`TransformersLmConfig`]: which model to load and how long a reply may run.

use js_sys::{Object, Reflect};
use wasm_bindgen::prelude::*;

/// How [`TransformersLm`](crate::TransformersLm) loads and drives a
/// Transformers.js text-generation pipeline.
///
/// Every field but `model` has a working default, so a Hub id is normally the
/// whole configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformersLmConfig {
    /// Hub id of an instruction-tuned model with a chat template, e.g.
    /// `"HuggingFaceTB/SmolLM2-360M-Instruct"`.
    pub model: String,
    /// Weight quantization: `"q8"`, `"q4f16"`, `"fp32"`, … `None` leaves the
    /// engine's own default, which picks per device.
    pub dtype: Option<String>,
    /// Backend to run on: `"wasm"` or `"webgpu"`. `None` leaves the engine's
    /// default.
    pub device: Option<String>,
    /// Called with the engine's own progress events while the model loads —
    /// one per file fetched, carrying `status`, `file`, `loaded`, and `total`.
    /// `None` reports nothing.
    pub progress: Option<js_sys::Function>,
    /// Tokens a reply may run to when [`GenParams`](pipecrab_lm::GenParams)
    /// sets no limit. Not left to the engine: its fallback is a 20-token
    /// `max_length` that counts the prompt, which ends a chat reply after one
    /// token.
    pub max_new_tokens: u32,
}

impl TransformersLmConfig {
    /// Load `model` from the Hub with replies of up to 256 tokens, leaving
    /// quantization and device to the engine.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            dtype: None,
            device: None,
            progress: None,
            max_new_tokens: 256,
        }
    }

    /// Options for `pipeline(...)`: how the model is loaded.
    pub(crate) fn pipeline_options(&self) -> Object {
        let options = Object::new();
        set(&options, "dtype", self.dtype.as_deref());
        set(&options, "device", self.device.as_deref());
        set(&options, "progress_callback", self.progress.clone());
        options
    }
}

/// Set `key` when the value is present. Fallible only for exotic receivers (a
/// proxy, a frozen object); a plain `Object` never rejects.
pub(crate) fn set(options: &Object, key: &str, value: Option<impl Into<JsValue>>) {
    if let Some(value) = value {
        let _ = Reflect::set(options, &JsValue::from_str(key), &value.into());
    }
}
