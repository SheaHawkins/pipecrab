//! [`WebAudioSource`]: microphone capture over the [`AudioSource`] trait.
//!
//! The worklet posts one mono chunk at a time to a JS callback, which pushes it
//! into a bounded queue; the async [`next_chunk`](AudioSource::next_chunk) pops
//! from that queue, parking until the next chunk lands.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::channel::mpsc;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use pipecrab_audio::{AudioChunk, AudioError, AudioFormat, AudioSource};

use crate::WebAudioConfig;

#[wasm_bindgen(module = "/js/capture.js")]
extern "C" {
    type Capture;

    #[wasm_bindgen(catch, js_name = openCapture)]
    fn open_capture(
        chunk_ms: u32,
        on_chunk: &Closure<dyn FnMut(Float32Array)>,
    ) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(method, getter, js_name = sampleRate)]
    fn sample_rate(this: &Capture) -> f64;

    #[wasm_bindgen(method)]
    fn close(this: &Capture);
}

/// Captures mono `f32` audio from the browser's microphone.
pub struct WebAudioSource {
    capture: Capture,
    chunks: mpsc::Receiver<Arc<[f32]>>,
    format: AudioFormat,
    overruns: Rc<Cell<usize>>,
    /// Keeps the worklet's message callback alive for the source's lifetime;
    /// dropping it would leave JS calling into freed Rust.
    _on_chunk: Closure<dyn FnMut(Float32Array)>,
}

impl WebAudioSource {
    /// Ask for microphone permission and start capturing, chunked per `config`.
    ///
    /// Call this from a user gesture (a click handler): browsers start an
    /// `AudioContext` suspended and refuse `getUserMedia` outside a secure
    /// context, and both surface here as [`AudioError::Device`].
    pub async fn open(config: &WebAudioConfig) -> Result<Self, AudioError> {
        let (sender, chunks) = mpsc::channel(config.queue_chunks);
        let sender = RefCell::new(sender);
        let overruns = Rc::new(Cell::new(0usize));
        let counter = overruns.clone();

        let on_chunk = Closure::<dyn FnMut(Float32Array)>::new(move |array: Float32Array| {
            let samples: Arc<[f32]> = array.to_vec().into();
            if sender.borrow_mut().try_send(samples).is_err() {
                counter.set(counter.get().saturating_add(1));
            }
        });

        let promise = open_capture(config.chunk_ms, &on_chunk).map_err(device_error)?;
        let capture: Capture = JsFuture::from(promise)
            .await
            .map_err(device_error)?
            .unchecked_into();

        let format = AudioFormat::new(capture.sample_rate() as u32, 1);
        Ok(Self {
            capture,
            chunks,
            format,
            overruns,
            _on_chunk: on_chunk,
        })
    }

    /// Chunks dropped so far because the queue was full (the pipeline was not
    /// popping fast enough). Monotonic; a healthy capture stays at 0.
    pub fn overruns(&self) -> usize {
        self.overruns.get()
    }
}

impl Drop for WebAudioSource {
    fn drop(&mut self) {
        // Stops the media tracks, so the browser's recording indicator clears.
        self.capture.close();
    }
}

// `?Send` unconditionally: this crate exists only on wasm32, where the trait's
// `maybe_async_trait!` resolves to exactly this.
#[async_trait(?Send)]
impl AudioSource for WebAudioSource {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>, AudioError> {
        Ok(self
            .chunks
            .next()
            .await
            .map(|samples| AudioChunk::new(samples, self.format)))
    }
}

/// Render a rejected promise or a thrown error as an [`AudioError::Device`].
pub(crate) fn device_error(error: JsValue) -> AudioError {
    let message = error
        .as_string()
        .or_else(|| {
            js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                .ok()
                .and_then(|m| m.as_string())
        })
        .unwrap_or_else(|| format!("{error:?}"));
    AudioError::Device(message)
}
