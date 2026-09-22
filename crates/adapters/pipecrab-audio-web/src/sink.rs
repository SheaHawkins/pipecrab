//! [`WebAudioSink`]: speaker playback over the [`AudioSink`] trait.
//!
//! Each chunk is scheduled on the Web Audio clock behind the last one, so
//! [`play`](AudioSink::play) never waits and [`cancel`](AudioSink::cancel) stops
//! everything scheduled at once.

use async_trait::async_trait;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use pipecrab_audio::{AudioChunk, AudioError, AudioFormat, AudioSink};

use crate::source::device_error;

#[wasm_bindgen(module = "/js/playback.js")]
extern "C" {
    type Playback;

    #[wasm_bindgen(catch, js_name = openPlayback)]
    fn open_playback() -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(method, getter, js_name = sampleRate)]
    fn sample_rate(this: &Playback) -> f64;

    #[wasm_bindgen(method, catch)]
    fn enqueue(this: &Playback, samples: &Float32Array) -> Result<(), JsValue>;

    #[wasm_bindgen(method)]
    fn cancel(this: &Playback);

    #[wasm_bindgen(method)]
    fn close(this: &Playback);
}

/// Plays mono `f32` audio through the browser's default output device.
///
/// No backpressure: the Web Audio graph holds every scheduled chunk, so the
/// queue is as deep as what the pipeline has produced. That is what keeps
/// playback gapless while inference occupies the main thread.
pub struct WebAudioSink {
    playback: Playback,
    format: AudioFormat,
}

impl WebAudioSink {
    /// Start an output context at the browser's own rate.
    ///
    /// Call this from a user gesture (a click handler): browsers start an
    /// `AudioContext` suspended, and resuming it needs one.
    pub async fn open() -> Result<Self, AudioError> {
        let promise = open_playback().map_err(device_error)?;
        let playback: Playback = JsFuture::from(promise)
            .await
            .map_err(device_error)?
            .unchecked_into();
        let format = AudioFormat::new(playback.sample_rate() as u32, 1);
        Ok(Self { playback, format })
    }
}

impl Drop for WebAudioSink {
    fn drop(&mut self) {
        self.playback.close();
    }
}

// `?Send` unconditionally: this crate exists only on wasm32, where the trait's
// `maybe_async_trait!` resolves to exactly this.
#[async_trait(?Send)]
impl AudioSink for WebAudioSink {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn play(&mut self, chunk: AudioChunk) -> Result<(), AudioError> {
        if chunk.format != self.format {
            return Err(AudioError::FormatMismatch {
                expected: self.format,
                got: chunk.format,
            });
        }
        // Web Audio has no zero-length buffer.
        if chunk.samples.is_empty() {
            return Ok(());
        }
        self.playback
            .enqueue(&Float32Array::from(&chunk.samples[..]))
            .map_err(device_error)
    }

    fn cancel(&mut self) {
        self.playback.cancel();
    }
}
