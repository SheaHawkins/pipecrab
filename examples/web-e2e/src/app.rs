//! The pipeline, and the two functions the page calls: [`start`] and
//! [`Session::stop`].
//!
//! ```text
//!   WebAudioSource (mic)
//!       ▼
//!   ResamplerStage (16 kHz mono)
//!       ▼
//!   VadStage<Debounced<SileroScorer>> ──▶ SttStage<Buffered<TransformersStt>>
//!       ▼
//!   BargeInStage               (speech onset ⇒ downstream Interrupt)
//!       ▼
//!   UserTurnGate
//!       ▼
//!   LmStage<TransformersLm>    (streams agent partials + one final)
//!       ▼
//!   SentenceChunker            (one final agent transcript per sentence)
//!       ▼
//!   AgentEcho                  (shows each sentence, forwards it)
//!       ▼
//!   TtsStage<KokoroTts>        (24 kHz mono audio per clause)
//!       ▼
//!   ResamplerStage (output rate) ──▶ WebAudioSink (speakers)
//! ```

use std::collections::VecDeque;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::channel::oneshot;
use pipecrab::{
    DataFrame, Decision, Direction, Finality, Outbound, PipelineBuilder, Processor, Received, Role,
    Stage, StageError, SystemFrame, Transcript,
};
use pipecrab_audio::{AudioFormat, AudioSink, AudioSource, ResamplerStage};
use pipecrab_audio_web::{WebAudioConfig, WebAudioSink, WebAudioSource};
use pipecrab_lm::LmStage;
use pipecrab_lm_web::{TransformersLm, TransformersLmConfig};
use pipecrab_stt::{Buffered, SttStage};
use pipecrab_stt_web::{TransformersStt, TransformersSttConfig};
use pipecrab_tts::{SentenceChunker, Synthesizer, TtsStage};
use pipecrab_tts_web::{KokoroTts, KokoroTtsConfig};
use pipecrab_turn::BargeInStage;
use pipecrab_vad::{DebounceConfig, Debounced, GateConfig, VadStage};
use pipecrab_vad_web::{SileroConfig, SileroScorer};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

/// What VAD and STT are fed: Silero's 16 kHz branch, which is also every ASR
/// export's rate. The browser captures at its own rate, so a resampler leads.
const ENGINE_FORMAT: AudioFormat = AudioFormat {
    sample_rate: 16_000,
    channels: 1,
};

const SYSTEM_PROMPT: &str = "You are a friendly voice assistant. Replies are \
     spoken aloud, so answer in one or two short sentences of plain prose with \
     no markup.";

/// Silero scores 32 ms windows. Eight of them before speech opens keeps a
/// cough from barging in on a reply; sixteen of silence before it closes
/// rides out a pause mid-sentence.
const DEBOUNCE: DebounceConfig = DebounceConfig {
    threshold: 0.5,
    start_windows: 8,
    stop_windows: 16,
};

/// Reports each completed user utterance to the page and drops empty finals so
/// a noise trigger never wakes the language model; every other frame forwards
/// untouched.
///
/// [`LmStage`] *consumes* user transcripts, so this stage — sitting just above
/// it — is the last place the conversation's user side can be observed.
struct UserTurnGate {
    on_event: js_sys::Function,
}

/// One completed user utterance to show.
struct ShowUser(Arc<str>);

impl Processor for UserTurnGate {
    type Effect = ShowUser;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<ShowUser> {
        match frame {
            DataFrame::Transcript(Transcript {
                role: Role::User,
                finality: Finality::Final,
                text,
            }) => match text.trim().is_empty() {
                // A noise trigger the model scored as blank: consume it so the
                // LM does not generate a reply to an empty turn.
                true => Decision::drop(),
                false => Decision::forward().emit(ShowUser(text.clone())),
            },
            _ => Decision::forward(),
        }
    }
}

#[async_trait(?Send)]
impl Stage for UserTurnGate {
    async fn perform(&self, ShowUser(text): ShowUser, _out: &Outbound) -> Result<(), StageError> {
        emit(&self.on_event, "user", &text);
        Ok(())
    }
}

/// Reports each completed agent sentence to the page on its way to synthesis;
/// the sentence itself forwards untouched.
///
/// [`TtsStage`] *consumes* final agent transcripts, so this stage — between the
/// chunker and the synthesizer — is the last place the conversation's agent
/// side can be observed as text.
struct AgentEcho {
    on_event: js_sys::Function,
}

/// One completed agent sentence to show.
struct ShowAgent(Arc<str>);

impl Processor for AgentEcho {
    type Effect = ShowAgent;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<ShowAgent> {
        match frame {
            DataFrame::Transcript(Transcript {
                role: Role::Agent,
                finality: Finality::Final,
                text,
            }) => Decision::forward().emit(ShowAgent(text.clone())),
            _ => Decision::forward(),
        }
    }
}

#[async_trait(?Send)]
impl Stage for AgentEcho {
    async fn perform(&self, ShowAgent(text): ShowAgent, _out: &Outbound) -> Result<(), StageError> {
        emit(&self.on_event, "agent", &text);
        Ok(())
    }
}

/// A running pipeline. Dropping it, or calling [`stop`](Session::stop), ends
/// capture, silences playback, and releases the microphone.
#[wasm_bindgen]
pub struct Session {
    /// Dropping the sender cancels the run: the pipeline future is dropped,
    /// which closes the head and cascades a clean shutdown downstream.
    cancel: Option<oneshot::Sender<()>>,
}

#[wasm_bindgen]
impl Session {
    /// Stop listening and speaking and shut the pipeline down. Idempotent.
    pub fn stop(&mut self) {
        self.cancel.take();
    }
}

/// Open the speakers and the microphone, load all four models, and run until
/// [`Session::stop`].
///
/// `ort`, `transformers`, and `kokoro` are the engine namespaces the page
/// imported — pipecrab's browser adapters take the engine rather than fetching
/// one, so the page keeps control of where its JS comes from. `options` names
/// the models: `vadModel` (a URL), `sttModel`, `lmModel`, and `ttsModel` (Hub
/// ids), `voice`, and the `device`, `lmDtype`, and `ttsDtype` the LM and TTS
/// run with.
///
/// `on_event` is called as `(kind, text)` with `kind` one of `"status"`,
/// `"speech"`, `"user"`, `"agent"`, `"latency"`, or `"error"`.
///
/// `on_progress` goes straight to the engines, which call it with their own
/// progress events — one per file fetched — while the three Hub models load.
///
/// Call this from a click handler: the microphone prompt and both
/// `AudioContext`s need a user gesture.
#[wasm_bindgen]
pub async fn start(
    ort: JsValue,
    transformers: JsValue,
    kokoro: JsValue,
    options: JsValue,
    on_event: js_sys::Function,
    on_progress: js_sys::Function,
) -> Result<Session, JsValue> {
    console_error_panic_hook::set_once();

    let device = option(&options, "device")?;

    // Opened first, while the click that started this still counts as a
    // gesture; everything after it awaits.
    emit(&on_event, "status", "opening the speakers");
    let sink = WebAudioSink::open().await.map_err(js_error)?;

    emit(&on_event, "status", "opening the microphone");
    let source = WebAudioSource::open(&WebAudioConfig::default())
        .await
        .map_err(js_error)?;

    emit(&on_event, "status", "loading the VAD model");
    let scorer = SileroScorer::load(&ort, SileroConfig::new(option(&options, "vadModel")?))
        .await
        .map_err(js_error)?;

    let stt_model = option(&options, "sttModel")?;
    emit(&on_event, "status", &format!("loading {stt_model}"));
    let mut stt_config = TransformersSttConfig::new(stt_model);
    stt_config.progress = Some(on_progress.clone());
    let transcriber = TransformersStt::load(&transformers, stt_config)
        .await
        .map_err(js_error)?;

    let lm_model = option(&options, "lmModel")?;
    emit(
        &on_event,
        "status",
        &format!("loading {lm_model} on {device}"),
    );
    let mut lm_config = TransformersLmConfig::new(lm_model);
    lm_config.dtype = Some(option(&options, "lmDtype")?);
    lm_config.device = Some(device.clone());
    lm_config.progress = Some(on_progress.clone());
    let lm = TransformersLm::load(&transformers, lm_config)
        .await
        .map_err(js_error)?;

    let tts_model = option(&options, "ttsModel")?;
    emit(
        &on_event,
        "status",
        &format!("loading {tts_model} on {device}"),
    );
    let mut tts_config = KokoroTtsConfig::new(tts_model);
    tts_config.voice = option(&options, "voice")?;
    tts_config.dtype = Some(option(&options, "ttsDtype")?);
    tts_config.device = Some(device.clone());
    tts_config.progress = Some(on_progress);
    let tts = KokoroTts::load(&kokoro, tts_config)
        .await
        .map_err(js_error)?;

    let status = format!(
        "listening — capturing at {} Hz, speaking at {} Hz, Kokoro at {} Hz",
        source.format().sample_rate,
        sink.format().sample_rate,
        tts.output_format().sample_rate,
    );

    let (ends, driver) = PipelineBuilder::new()
        .stage(ResamplerStage::new(ENGINE_FORMAT).map_err(js_error)?)
        .stage(VadStage::with_config(
            Debounced::with_config(scorer, DEBOUNCE),
            GateConfig {
                preroll: Duration::from_secs(1),
            },
        ))
        .stage(SttStage::new(Buffered::new(transcriber)))
        .stage(BargeInStage::new())
        .stage(UserTurnGate {
            on_event: on_event.clone(),
        })
        .stage(LmStage::new(lm, SYSTEM_PROMPT))
        .stage(SentenceChunker::new())
        .stage(AgentEcho {
            on_event: on_event.clone(),
        })
        .stage(TtsStage::new(tts))
        .stage(ResamplerStage::new(sink.format()).map_err(js_error)?)
        .build()
        .start();
    let input = ends.input;
    let mut output = ends.output;

    emit(&on_event, "status", &status);

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

    // `play` only schedules, so unlike the desktop pump this one never parks on
    // a push: it reads the tail's system lane as promptly as its data lane, and
    // needs no race to see a barge-in.
    let pump_events = on_event.clone();
    let pump_out = async move {
        let mut sink = sink;
        // Keepers from the tail-lane flush on a barge-in, replayed ahead of the
        // next receive.
        let mut pending: VecDeque<DataFrame> = VecDeque::new();
        // When the user last stopped speaking, until the reply's first audio.
        let mut stopped_at = None;
        loop {
            let received = match pending.pop_front() {
                Some(frame) => Received::Data(frame),
                None => match output.recv().await {
                    Some(received) => received,
                    None => break,
                },
            };
            match received {
                Received::Data(DataFrame::Audio(chunk)) => {
                    if let Some(stopped) = stopped_at.take() {
                        let waited = js_sys::Date::now() - stopped;
                        emit(&pump_events, "latency", &format!("{waited:.0}"));
                    }
                    if let Err(error) = sink.play(chunk).await {
                        emit(&pump_events, "error", &format!("playback stopped: {error}"));
                        break;
                    }
                }
                Received::Data(DataFrame::SpeechStarted) => {
                    stopped_at = None;
                    emit(&pump_events, "speech", "started");
                }
                Received::Data(DataFrame::SpeechStopped) => {
                    stopped_at = Some(js_sys::Date::now());
                    emit(&pump_events, "speech", "stopped");
                }
                // Nothing flushes the tail lane for the application, so this is
                // where the agent stops talking: silence what is scheduled and
                // drop the agent audio still queued past the tail.
                Received::Sys(_, SystemFrame::Interrupt) => {
                    sink.cancel();
                    // Held keepers predate this interrupt: re-apply the flush
                    // predicate before collecting the new keepers.
                    pending.retain(|frame| frame.survives_flush());
                    pending.extend(output.flush_data());
                }
                Received::Sys(_, SystemFrame::Error { message, fatal: _ }) => {
                    emit(&pump_events, "error", &message);
                }
                Received::Data(_) | Received::Sys(_, _) => {}
            }
        }
    };

    let (cancel, cancelled) = oneshot::channel();
    spawn_local(async move {
        let run = async { futures::join!(driver, pump_in, pump_out) };
        futures::pin_mut!(run);
        // Whichever finishes first wins; dropping `run` releases the microphone
        // and the speakers along with everything else the pipeline owns.
        let _ = futures::future::select(run, cancelled).await;
        emit(&on_event, "status", "stopped");
    });

    Ok(Session {
        cancel: Some(cancel),
    })
}

/// Read one string field of the page's `options`.
fn option(options: &JsValue, key: &str) -> Result<String, JsValue> {
    js_sys::Reflect::get(options, &JsValue::from_str(key))?
        .as_string()
        .ok_or_else(|| js_sys::Error::new(&format!("options.{key} must be a string")).into())
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
