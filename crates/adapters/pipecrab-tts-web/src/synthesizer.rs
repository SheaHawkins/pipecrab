//! [`KokoroTts`]: one kokoro-js model behind [`Synthesizer`].

use async_trait::async_trait;
use futures::StreamExt;
use js_sys::{Float32Array, Object, Reflect};
use pipecrab_core::{AudioChunk, AudioFormat};
use pipecrab_tts::{Synthesizer, TtsAudioStream, TtsError};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::KokoroTtsConfig;
use crate::clauses::{clauses, trim_joins};

/// Every Kokoro voice speaks 24 kHz mono.
const FORMAT: AudioFormat = AudioFormat {
    sample_rate: 24_000,
    channels: 1,
};

#[wasm_bindgen(module = "/js/kokoro.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = loadKokoro)]
    fn load_kokoro(
        kokoro: &JsValue,
        model: &str,
        voice: &str,
        options: &Object,
    ) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(catch, js_name = synthesize)]
    fn synthesize_js(
        handle: &JsValue,
        text: &str,
        options: &Object,
    ) -> Result<js_sys::Promise, JsValue>;
}

/// Speaks text with Kokoro on kokoro-js, a clause at a time.
///
/// Kokoro renders its whole input in one inference run, which nothing can stop
/// part-way. So each sentence is split at its clause punctuation and each
/// clause is its own run and its own chunk: speech starts after the first
/// clause, and a barge-in waits out at most the clause in flight — the rest of
/// the sentence never starts.
pub struct KokoroTts {
    /// The JS-side model and its queue of synthesis runs.
    handle: JsValue,
    speak_options: Object,
}

impl KokoroTts {
    /// Fetch and warm the model named by `config` on the page's kokoro-js.
    ///
    /// `kokoro` is the namespace the page imported. The model download and its
    /// compilation both happen here, so the wait — about a hundred megabytes on
    /// a cold cache — is paid at construction rather than on the first reply.
    ///
    /// # Errors
    ///
    /// Returns [`TtsError::Engine`] if the model cannot be fetched, if the
    /// engine rejects the requested device or quantization, or if the model has
    /// no voice by the configured name.
    pub async fn load(kokoro: &JsValue, config: KokoroTtsConfig) -> Result<Self, TtsError> {
        let promise = load_kokoro(kokoro, &config.model, &config.voice, &config.load_options())
            .map_err(engine_error)?;
        let handle = JsFuture::from(promise).await.map_err(engine_error)?;
        Ok(Self {
            handle,
            speak_options: config.speak_options(),
        })
    }
}

// `?Send` unconditionally: this crate exists only on wasm32, where the trait is
// declared exactly this way.
#[async_trait(?Send)]
impl Synthesizer for KokoroTts {
    fn output_format(&self) -> AudioFormat {
        FORMAT
    }

    async fn synthesize(&self, text: &str) -> Result<TtsAudioStream, TtsError> {
        let handle = self.handle.clone();
        let options = self.speak_options.clone();
        let clauses: Vec<String> = clauses(text).into_iter().map(str::to_owned).collect();
        let count = clauses.len();
        // `then` starts a clause's run only when the stage pulls for it, so a
        // dropped stream leaves the clauses after it unspoken.
        let stream = futures::stream::iter(clauses.into_iter().enumerate())
            .then(move |(at, clause)| {
                let joins = (at > 0, at + 1 < count);
                speak(handle.clone(), options.clone(), clause, joins)
            })
            .filter_map(|chunk| futures::future::ready(chunk.transpose()));
        Ok(Box::pin(stream))
    }

    /// Nothing to stop: the clause in flight is one inference run, which cannot
    /// be aborted part-way. The stage drops the stream, so its audio is never
    /// emitted and the clauses after it never start; the next synthesis queues
    /// behind the one run still going.
    fn cancel(&self) {}
}

/// Speak one clause, its edges trimmed where `joins` says it meets a
/// neighbour; `None` if the model returned no audio for it.
async fn speak(
    handle: JsValue,
    options: Object,
    clause: String,
    (joined_before, joined_after): (bool, bool),
) -> Result<Option<AudioChunk>, TtsError> {
    let promise = synthesize_js(&handle, &clause, &options).map_err(engine_error)?;
    let audio = JsFuture::from(promise).await.map_err(engine_error)?;

    let rate = Reflect::get(&audio, &JsValue::from_str("sampleRate"))
        .ok()
        .and_then(|rate| rate.as_f64());
    if rate != Some(f64::from(FORMAT.sample_rate)) {
        return Err(TtsError::Engine(format!(
            "model spoke at {rate:?} Hz, not {} Hz",
            FORMAT.sample_rate
        )));
    }
    let samples: Float32Array = Reflect::get(&audio, &JsValue::from_str("samples"))
        .ok()
        .and_then(|samples| samples.dyn_into().ok())
        .ok_or_else(|| TtsError::Engine("model returned no samples".into()))?;

    let samples = samples.to_vec();
    let kept = trim_joins(&samples, joined_before, joined_after);
    Ok((!kept.is_empty()).then(|| AudioChunk::new(kept.into(), FORMAT)))
}

/// Render a rejected promise or a thrown error as a [`TtsError::Engine`].
fn engine_error(error: JsValue) -> TtsError {
    let message = error
        .as_string()
        .or_else(|| {
            js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                .ok()
                .and_then(|m| m.as_string())
        })
        .unwrap_or_else(|| format!("{error:?}"));
    TtsError::Engine(message)
}
