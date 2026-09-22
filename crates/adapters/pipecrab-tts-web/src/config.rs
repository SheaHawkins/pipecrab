//! [`KokoroTtsConfig`]: which model and voice to load.

use js_sys::{Object, Reflect};
use wasm_bindgen::prelude::*;

/// How [`KokoroTts`](crate::KokoroTts) loads and drives kokoro-js.
///
/// Every field but `model` has a working default, so a Hub id is normally the
/// whole configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct KokoroTtsConfig {
    /// Hub id of a Kokoro ONNX export, e.g. `"onnx-community/Kokoro-82M-v1.0-ONNX"`.
    pub model: String,
    /// Voice to speak in, e.g. `"af_heart"`. Checked against the model's voices
    /// at load.
    pub voice: String,
    /// Speaking rate; `1.0` is the voice's natural pace.
    pub speed: f32,
    /// Weight quantization: `"q8"`, `"fp16"`, `"fp32"`, … `None` leaves the
    /// engine's own default.
    pub dtype: Option<String>,
    /// Backend to run on: `"wasm"` or `"webgpu"`. `None` leaves the engine's
    /// default.
    pub device: Option<String>,
    /// Called with the engine's own progress events while the model loads —
    /// one per file fetched, carrying `status`, `file`, `loaded`, and `total`.
    /// `None` reports nothing.
    pub progress: Option<js_sys::Function>,
}

impl KokoroTtsConfig {
    /// Load `model` from the Hub and speak as `af_heart` at natural pace,
    /// leaving quantization and device to the engine.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            voice: "af_heart".into(),
            speed: 1.0,
            dtype: None,
            device: None,
            progress: None,
        }
    }

    /// Options for `from_pretrained(...)`: how the model is loaded.
    pub(crate) fn load_options(&self) -> Object {
        let options = Object::new();
        set(&options, "dtype", self.dtype.as_deref());
        set(&options, "device", self.device.as_deref());
        set(&options, "progress_callback", self.progress.clone());
        options
    }

    /// Options for each `generate(...)`: how the text is spoken.
    pub(crate) fn speak_options(&self) -> Object {
        let options = Object::new();
        set(&options, "voice", Some(self.voice.as_str()));
        set(&options, "speed", Some(self.speed));
        options
    }
}

/// Set `key` when the value is present. Fallible only for exotic receivers (a
/// proxy, a frozen object); a plain `Object` never rejects.
fn set(options: &Object, key: &str, value: Option<impl Into<JsValue>>) {
    if let Some(value) = value {
        let _ = Reflect::set(options, &JsValue::from_str(key), &value.into());
    }
}
