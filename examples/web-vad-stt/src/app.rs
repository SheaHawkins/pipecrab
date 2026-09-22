//! The pipeline, and the two functions the page calls: [`start`] and
//! [`Session::stop`].

use std::error::Error;

use futures::channel::oneshot;
use pipecrab::{DataFrame, Direction, Finality, PipelineBuilder, Received, SystemFrame};
use pipecrab_audio::{AudioFormat, AudioSource, ResamplerStage};
use pipecrab_audio_web::{WebAudioConfig, WebAudioSource};
use pipecrab_stt::{Buffered, SttStage};
use pipecrab_stt_web::{TransformersStt, TransformersSttConfig};
use pipecrab_vad::{Debounced, VadStage};
use pipecrab_vad_web::{SileroConfig, SileroScorer};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

/// What both engines are fed: Silero's 16 kHz branch, which is also every ASR
/// export's rate. The browser captures at its own rate, so a resampler leads.
const ENGINE_FORMAT: AudioFormat = AudioFormat {
    sample_rate: 16_000,
    channels: 1,
};

/// A running pipeline. Dropping it, or calling [`stop`](Session::stop), ends
/// capture and releases the microphone.
#[wasm_bindgen]
pub struct Session {
    /// Dropping the sender cancels the run: the pipeline future is dropped,
    /// which closes the head and cascades a clean shutdown downstream.
    cancel: Option<oneshot::Sender<()>>,
}

#[wasm_bindgen]
impl Session {
    /// Stop capturing and shut the pipeline down. Idempotent.
    pub fn stop(&mut self) {
        self.cancel.take();
    }
}

/// Open the microphone, load both models, and run until [`Session::stop`].
///
/// `ort` and `transformers` are the engine namespaces the page imported —
/// pipecrab's browser adapters take the engine rather than fetching one, so the
/// page keeps control of where its JS comes from.
///
/// `on_event` is called as `(kind, text)` with `kind` one of `"status"`,
/// `"speech"`, `"partial"`, `"final"`, or `"error"`.
///
/// `on_progress` goes straight to the engine, which calls it with its own
/// progress events — one per file fetched — while the model loads.
///
/// Call this from a click handler: the microphone prompt and the `AudioContext`
/// both need a user gesture.
#[wasm_bindgen]
pub async fn start(
    ort: JsValue,
    transformers: JsValue,
    vad_model_url: String,
    stt_model: String,
    on_event: js_sys::Function,
    on_progress: js_sys::Function,
) -> Result<Session, JsValue> {
    console_error_panic_hook::set_once();

    emit(&on_event, "status", "opening the microphone");
    let source = WebAudioSource::open(&WebAudioConfig::default())
        .await
        .map_err(js_error)?;

    emit(&on_event, "status", "loading the VAD model");
    let scorer = SileroScorer::load(&ort, SileroConfig::new(vad_model_url))
        .await
        .map_err(js_error)?;

    emit(&on_event, "status", &format!("loading {stt_model}"));
    let mut stt_config = TransformersSttConfig::new(stt_model);
    stt_config.progress = Some(on_progress);
    let transcriber = TransformersStt::load(&transformers, stt_config)
        .await
        .map_err(js_error)?;

    // Silero scores 512-sample windows and `Debounced` turns those probabilities
    // into speech edges; `VadStage` gates on the edges, so only utterance audio —
    // pre-roll included — ever reaches the transcriber. `Buffered` collects that
    // utterance and transcribes it once, at its end.
    let (ends, driver) = PipelineBuilder::new()
        .stage(ResamplerStage::new(ENGINE_FORMAT).map_err(js_error)?)
        .stage(VadStage::new(Debounced::new(scorer)))
        .stage(SttStage::new(Buffered::new(transcriber)))
        .build()
        .start();
    let input = ends.input;
    let mut output = ends.output;

    emit(
        &on_event,
        "status",
        &format!(
            "listening — capturing at {} Hz, processing at {} Hz mono",
            source.format().sample_rate,
            ENGINE_FORMAT.sample_rate,
        ),
    );

    let pump_events = on_event.clone();
    let pump_in = async move {
        let mut source = source;
        let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
        loop {
            match source.next_chunk().await {
                Ok(Some(chunk)) => {
                    if input.send_data(DataFrame::Audio(chunk)).await.is_err() {
                        break; // downstream gone
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    emit(&pump_events, "error", &format!("capture stopped: {error}"));
                    break;
                }
            }
        }
        // `input` is dropped here → the pipeline shuts down.
    };

    let drain_events = on_event.clone();
    let drain = async move {
        while let Some(received) = output.recv().await {
            match received {
                Received::Data(DataFrame::SpeechStarted) => {
                    emit(&drain_events, "speech", "started");
                }
                Received::Data(DataFrame::SpeechStopped) => {
                    emit(&drain_events, "speech", "stopped");
                }
                Received::Data(DataFrame::Transcript(transcript)) => match transcript.finality {
                    Finality::Partial { .. } => {
                        emit(&drain_events, "partial", &transcript.text);
                    }
                    Finality::Final if transcript.text.is_empty() => {
                        emit(&drain_events, "final", "<no speech recognized>");
                    }
                    Finality::Final => emit(&drain_events, "final", &transcript.text),
                },
                Received::Sys(_, SystemFrame::Error { message, fatal: _ }) => {
                    emit(&drain_events, "error", &message);
                }
                Received::Data(_) | Received::Sys(_, _) => {}
            }
        }
    };

    let (cancel, cancelled) = oneshot::channel();
    spawn_local(async move {
        let run = async { futures::join!(driver, pump_in, drain) };
        futures::pin_mut!(run);
        // Whichever finishes first wins; dropping `run` releases the microphone
        // along with everything else the pipeline owns.
        let _ = futures::future::select(run, cancelled).await;
        emit(&on_event, "status", "stopped");
    });

    Ok(Session {
        cancel: Some(cancel),
    })
}

/// Report one line to the page.
fn emit(on_event: &js_sys::Function, kind: &str, text: &str) {
    let _ = on_event.call2(
        &JsValue::NULL,
        &JsValue::from_str(kind),
        &JsValue::from_str(text),
    );
}

/// Surface a setup failure to JS as a thrown `Error`.
fn js_error(error: impl Error) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}
