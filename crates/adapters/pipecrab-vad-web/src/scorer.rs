//! [`SileroScorer`]: one onnxruntime-web session behind [`SpeechScorer`].

use std::sync::Arc;

use async_trait::async_trait;
use js_sys::Float32Array;
use pipecrab_core::AudioFormat;
use pipecrab_vad::{SpeechScorer, VadError};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::SileroConfig;

#[wasm_bindgen(module = "/js/silero.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = loadSilero)]
    fn load_silero(
        ort: &JsValue,
        model_url: &str,
        sample_rate: u32,
    ) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(catch, js_name = scoreSilero)]
    fn score_silero(handle: &JsValue, window: &Float32Array) -> Result<js_sys::Promise, JsValue>;
}

/// Scores 512-sample windows with Silero VAD on onnxruntime-web.
///
/// One session, one recurrent state, driven strictly in order — the pipeline
/// awaits each [`score`](SpeechScorer::score) before feeding the next window, so
/// the state carried between calls is never raced.
pub struct SileroScorer {
    /// The JS-side session and its carried recurrent state.
    handle: JsValue,
    format: AudioFormat,
    window: usize,
}

impl SileroScorer {
    /// Fetch and compile the model named by `config` on the page's `ort`.
    ///
    /// `ort` is the onnxruntime-web namespace the page imported. Fetching and
    /// compiling both happen here, so a slow first load is paid at construction
    /// rather than on the first window.
    ///
    /// # Errors
    ///
    /// Returns [`VadError::Engine`] if the model cannot be fetched or compiled,
    /// or if it is not a Silero VAD export (its recurrent inputs and outputs
    /// have to pair up).
    pub async fn load(ort: &JsValue, config: SileroConfig) -> Result<Self, VadError> {
        let promise =
            load_silero(ort, &config.model_url, config.sample_rate).map_err(engine_error)?;
        let handle = JsFuture::from(promise).await.map_err(engine_error)?;
        Ok(Self {
            handle,
            format: config.format(),
            window: config.window,
        })
    }
}

// `?Send` unconditionally: this crate exists only on wasm32, where the trait is
// declared exactly this way.
#[async_trait(?Send)]
impl SpeechScorer for SileroScorer {
    fn input_format(&self) -> AudioFormat {
        self.format
    }

    fn window_len(&self) -> usize {
        self.window
    }

    async fn score(&self, window: Arc<[f32]>) -> Result<f32, VadError> {
        // Copied into JS memory rather than viewed: a view into the wasm heap is
        // invalidated by any allocation the awaited inference makes.
        let samples = Float32Array::from(&window[..]);
        let promise = score_silero(&self.handle, &samples).map_err(engine_error)?;
        let probability = JsFuture::from(promise).await.map_err(engine_error)?;
        probability.as_f64().map(|p| p as f32).ok_or_else(|| {
            VadError::Engine(format!("model returned {probability:?}, not a number"))
        })
    }
}

/// Render a rejected promise or a thrown error as a [`VadError::Engine`].
fn engine_error(error: JsValue) -> VadError {
    let message = error
        .as_string()
        .or_else(|| {
            js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                .ok()
                .and_then(|m| m.as_string())
        })
        .unwrap_or_else(|| format!("{error:?}"));
    VadError::Engine(message)
}
