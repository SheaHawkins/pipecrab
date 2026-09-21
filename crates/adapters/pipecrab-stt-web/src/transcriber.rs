//! [`TransformersStt`]: one Transformers.js pipeline behind [`Transcriber`].

use std::sync::Arc;

use async_trait::async_trait;
use js_sys::{Float32Array, Object};
use pipecrab_core::AudioFormat;
use pipecrab_stt::{SttError, Transcriber};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::TransformersSttConfig;

#[wasm_bindgen(module = "/js/transformers.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = loadTranscriber)]
    fn load_transcriber(
        transformers: &JsValue,
        model: &str,
        options: &Object,
    ) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(catch, js_name = transcribe)]
    fn transcribe_js(
        handle: &JsValue,
        samples: &Float32Array,
        options: &Object,
    ) -> Result<js_sys::Promise, JsValue>;
}

/// Transcribes an utterance with a Transformers.js ASR pipeline.
pub struct TransformersStt {
    /// The JS-side pipeline.
    handle: JsValue,
    format: AudioFormat,
    decode_options: Object,
}

impl TransformersStt {
    /// Fetch and warm the pipeline named by `config` on the page's Transformers.js.
    ///
    /// `transformers` is the namespace the page imported. The model download and
    /// its compilation both happen here, so the wait — tens of megabytes on a
    /// cold cache — is paid at construction rather than on the first utterance.
    ///
    /// # Errors
    ///
    /// Returns [`SttError::Engine`] if the model cannot be fetched, or if the
    /// engine rejects the requested device or quantization.
    pub async fn load(
        transformers: &JsValue,
        config: TransformersSttConfig,
    ) -> Result<Self, SttError> {
        let promise = load_transcriber(transformers, &config.model, &config.pipeline_options())
            .map_err(engine_error)?;
        let handle = JsFuture::from(promise).await.map_err(engine_error)?;
        Ok(Self {
            handle,
            format: config.format(),
            decode_options: config.decode_options(),
        })
    }
}

// `?Send` unconditionally: this crate exists only on wasm32, where the trait is
// declared exactly this way.
#[async_trait(?Send)]
impl Transcriber for TransformersStt {
    fn input_format(&self) -> AudioFormat {
        self.format
    }

    async fn transcribe(&self, samples: Arc<[f32]>) -> Result<String, SttError> {
        // Copied into JS memory rather than viewed: a view into the wasm heap is
        // invalidated by any allocation the awaited inference makes.
        let audio = Float32Array::from(&samples[..]);
        let promise =
            transcribe_js(&self.handle, &audio, &self.decode_options).map_err(engine_error)?;
        let text = JsFuture::from(promise).await.map_err(engine_error)?;
        text.as_string()
            .ok_or_else(|| SttError::Engine(format!("model returned {text:?}, not a string")))
    }
}

/// Render a rejected promise or a thrown error as an [`SttError::Engine`].
fn engine_error(error: JsValue) -> SttError {
    let message = error
        .as_string()
        .or_else(|| {
            js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                .ok()
                .and_then(|m| m.as_string())
        })
        .unwrap_or_else(|| format!("{error:?}"));
    SttError::Engine(message)
}
