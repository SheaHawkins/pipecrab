//! A full local voice agent, end to end: Sherpa VAD brackets each utterance
//! from the microphone, Moonshine v2 transcribes it, a llama.cpp chat model
//! streams a reply, and Kokoro speaks that reply sentence by sentence.
//!
//! ```text
//!   CpalSource (mic)
//!       ▼
//!   ResamplerStage (16 kHz mono)
//!       ▼
//!   VadStage<SherpaVad> ──▶ SttStage<OfflineSherpaStt> ──▶ UserTurnGate
//!       ▼
//!   LmStage<LlamaCpp>          (streams agent partials + one final)
//!       ▼
//!   SentenceChunker            (one final agent transcript per sentence)
//!       ▼
//!   AgentEcho                  (prints each sentence, forwards it)
//!       ▼
//!   TtsStage<SherpaTts>        (24 kHz mono audio per sentence)
//!       ▼
//!   ResamplerStage (device rate) ──▶ CpalSink (speaker)
//! ```
//!
//! The chunker is why the agent starts talking before the model has finished
//! writing: each completed sentence is synthesized while the next is still
//! being generated.
//!
//! **Use headphones** — over speakers the microphone re-captures the agent's
//! own voice and it talks to itself.

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
fn main() {
    if let Err(error) = desktop::run() {
        eprintln!("e2e-voice-agent: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("e2e-voice-agent requires a desktop OS (macOS, Windows, or Linux)");
    std::process::exit(1);
}

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
mod desktop {
    use std::error::Error;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use futures::executor::block_on;
    use pipecrab::{
        DataFrame, Decision, Direction, Finality, Outbound, PipelineBuilder, Processor, Received,
        Role, Stage, StageError, SystemFrame, Transcript,
    };
    use pipecrab_audio::{AudioFormat, AudioSink, AudioSource, ResamplerStage};
    use pipecrab_audio_cpal::{CpalConfig, CpalSink, CpalSource};
    use pipecrab_lm::LmStage;
    use pipecrab_lm_llamacpp::{LlamaCpp, LlamaCppConfig};
    use pipecrab_stt::SttStage;
    use pipecrab_stt_sherpa::{MoonshineV2Config, OfflineSherpaStt};
    use pipecrab_tts::{SentenceChunker, Synthesizer, TtsStage};
    use pipecrab_tts_sherpa::{KokoroConfig, SherpaTts};
    use pipecrab_vad::{GateConfig, VadStage};
    use pipecrab_vad_sherpa::{SherpaVad, SherpaVadConfig};

    const SHERPA_FORMAT: AudioFormat = AudioFormat {
        sample_rate: 16_000,
        channels: 1,
    };

    const DEFAULT_SYSTEM_PROMPT: &str = "You are a friendly voice assistant. \
         Replies are spoken aloud, so answer in one or two short sentences of \
         plain prose with no markup.";

    /// Prints each completed user utterance and drops empty finals so a noise
    /// trigger never wakes the language model; every other frame forwards
    /// untouched.
    ///
    /// [`LmStage`] *consumes* user transcripts, so this stage — sitting just
    /// above it — is the last place the conversation's user side can be
    /// observed. Following the [`Processor`]/[`Stage`] split, `decide_data`
    /// only picks the text out; the printing (I/O) happens in `perform`.
    struct UserTurnGate;

    /// One completed user utterance to print.
    struct PrintUser(Arc<str>);

    impl Processor for UserTurnGate {
        type Effect = PrintUser;

        fn decide_data(&mut self, frame: &DataFrame) -> Decision<PrintUser> {
            match frame {
                DataFrame::Transcript(Transcript {
                    role: Role::User,
                    finality: Finality::Final,
                    text,
                }) => match text.trim().is_empty() {
                    // A noise trigger the model scored as blank: consume it so
                    // the LM does not generate a reply to an empty turn.
                    true => Decision::drop(),
                    false => Decision::forward().emit(PrintUser(text.clone())),
                },
                _ => Decision::forward(),
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl Stage for UserTurnGate {
        async fn perform(
            &self,
            PrintUser(text): PrintUser,
            _out: &Outbound,
        ) -> Result<(), StageError> {
            println!("You: {text}");
            Ok(())
        }
    }

    /// Prints each completed agent sentence on its way to synthesis; the
    /// sentence itself forwards untouched.
    ///
    /// [`TtsStage`] *consumes* final agent transcripts, so this stage — between
    /// the chunker and the synthesizer — is the last place the conversation's
    /// agent side can be observed as text.
    struct AgentEcho;

    /// One completed agent sentence to print.
    struct PrintAgent(Arc<str>);

    impl Processor for AgentEcho {
        type Effect = PrintAgent;

        fn decide_data(&mut self, frame: &DataFrame) -> Decision<PrintAgent> {
            match frame {
                DataFrame::Transcript(Transcript {
                    role: Role::Agent,
                    finality: Finality::Final,
                    text,
                }) => Decision::forward().emit(PrintAgent(text.clone())),
                _ => Decision::forward(),
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl Stage for AgentEcho {
        async fn perform(
            &self,
            PrintAgent(text): PrintAgent,
            _out: &Outbound,
        ) -> Result<(), StageError> {
            println!("Agent: {text}");
            Ok(())
        }
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let Args {
            vad_model,
            encoder,
            merged_decoder,
            tokens,
            lm_model,
            tts_model,
            tts_voices,
            tts_tokens,
            tts_data_dir,
            speaker,
            speed,
            system_prompt,
            stt_threads,
            seconds,
        } = parse_args()?;

        let audio_config = CpalConfig::default();
        let source = CpalSource::new(&audio_config)?;
        let sink = CpalSink::new(&audio_config)?;
        let mut vad_config = SherpaVadConfig::new(vad_model);
        vad_config.threshold = 0.35;
        vad_config.min_speech_duration = 0.1;
        vad_config.min_silence_duration = 0.5;
        vad_config.max_speech_duration = 30.0;
        let detector = SherpaVad::new(vad_config)?;
        let mut stt_config = MoonshineV2Config::new(encoder, merged_decoder, tokens);
        stt_config.num_threads = stt_threads;
        let transcriber = OfflineSherpaStt::new(stt_config)?;
        let capture_resampler = ResamplerStage::new(SHERPA_FORMAT)?;
        let playback_resampler = ResamplerStage::new(sink.format())?;

        println!("e2e-voice-agent: loading {} …", lm_model.display());
        let model = LlamaCpp::load(LlamaCppConfig::new(lm_model))?;

        let mut tts_config = KokoroConfig::new(tts_model, tts_voices, tts_tokens, tts_data_dir);
        tts_config.speaker = speaker;
        tts_config.speed = speed;
        let synth = SherpaTts::new(tts_config)?;

        println!(
            "e2e-voice-agent: input = {} @ {} Hz mono",
            source.device_name(),
            source.format().sample_rate
        );
        println!(
            "e2e-voice-agent: output = {} @ {} Hz",
            sink.device_name(),
            sink.format().sample_rate
        );
        println!(
            "e2e-voice-agent: kokoro @ {} Hz mono, speaker {speaker}, speed {speed}",
            synth.output_format().sample_rate
        );
        println!("e2e-voice-agent: use headphones so the agent does not hear itself");
        match seconds {
            Some(seconds) => println!("e2e-voice-agent: listening for {seconds} seconds"),
            None => println!("e2e-voice-agent: listening until Ctrl-C"),
        }

        // The sink's ring holds up to this much queued audio; linger that long
        // after the pipeline closes so the spoken tail is not clipped.
        let ring_ms = u64::from(audio_config.chunk_ms) * audio_config.ring_chunks as u64;
        let max_chunks =
            seconds.map(|seconds| (seconds * 1000 / u64::from(audio_config.chunk_ms)) as usize);
        let (ends, driver) = PipelineBuilder::new()
            .stage(capture_resampler)
            .stage(VadStage::with_config(
                detector,
                GateConfig {
                    preroll: Duration::from_secs(1),
                },
            ))
            .stage(SttStage::new(transcriber))
            .stage(UserTurnGate)
            .stage(LmStage::new(model, system_prompt))
            .stage(SentenceChunker::new())
            .stage(AgentEcho)
            .stage(TtsStage::new(synth))
            .stage(playback_resampler)
            .build()
            .start();
        let input = ends.input;
        let mut output = ends.output;

        let pump_in = async move {
            let mut source = source;
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let mut sent = 0usize;
            loop {
                match source.next_chunk().await {
                    Ok(Some(chunk)) => {
                        if input.send_data(DataFrame::Audio(chunk)).await.is_err() {
                            break;
                        }
                        sent += 1;
                        if max_chunks.is_some_and(|maximum| sent >= maximum) {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        eprintln!("e2e-voice-agent: capture stopped: {error}");
                        break;
                    }
                }
            }
        };

        let pump_out = async move {
            let mut sink = sink;
            let mut utterance_started = None;
            while let Some(received) = output.recv().await {
                match received {
                    Received::Data(DataFrame::Audio(chunk)) => {
                        if let Err(error) = sink.play(chunk).await {
                            eprintln!("e2e-voice-agent: playback stopped: {error}");
                            break;
                        }
                    }
                    Received::Data(DataFrame::SpeechStarted) => {
                        utterance_started = Some(Instant::now());
                        println!("SpeechStarted");
                    }
                    Received::Data(DataFrame::SpeechStopped) => {
                        let elapsed = utterance_started.take().map(|started| started.elapsed());
                        match elapsed {
                            Some(elapsed) => {
                                println!("SpeechStopped ({:.2} s)", elapsed.as_secs_f32());
                            }
                            None => println!("SpeechStopped"),
                        }
                    }
                    Received::Sys(_, SystemFrame::Error { message, fatal: _ }) => {
                        eprintln!("e2e-voice-agent: pipeline error: {message}")
                    }
                    Received::Data(_) | Received::Sys(_, _) => {}
                }
            }
            // Everything is enqueued but the device is still draining its ring;
            // dropping the sink now would clip the agent's last words.
            std::thread::sleep(Duration::from_millis(ring_ms + 50));
        };

        block_on(async { futures::join!(driver, pump_in, pump_out) });
        println!("e2e-voice-agent: stopped");
        Ok(())
    }

    struct Args {
        vad_model: PathBuf,
        encoder: PathBuf,
        merged_decoder: PathBuf,
        tokens: PathBuf,
        lm_model: PathBuf,
        tts_model: PathBuf,
        tts_voices: PathBuf,
        tts_tokens: PathBuf,
        tts_data_dir: PathBuf,
        speaker: i32,
        speed: f32,
        system_prompt: String,
        stt_threads: i32,
        seconds: Option<u64>,
    }

    fn parse_args() -> Result<Args, String> {
        let mut vad_model = None;
        let mut encoder = None;
        let mut merged_decoder = None;
        let mut tokens = None;
        let mut lm_model = None;
        let mut tts_model = None;
        let mut tts_voices = None;
        let mut tts_tokens = None;
        let mut tts_data_dir = None;
        let mut speaker = 0;
        let mut speed = 1.0;
        let mut system_prompt = None;
        let mut stt_threads = 2;
        let mut seconds = None;
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--vad-model" => {
                    vad_model = Some(PathBuf::from(value(&mut args, "--vad-model")?));
                }
                "--encoder" => encoder = Some(PathBuf::from(value(&mut args, "--encoder")?)),
                "--merged-decoder" => {
                    merged_decoder = Some(PathBuf::from(value(&mut args, "--merged-decoder")?));
                }
                "--tokens" => tokens = Some(PathBuf::from(value(&mut args, "--tokens")?)),
                "--lm-model" => lm_model = Some(PathBuf::from(value(&mut args, "--lm-model")?)),
                "--tts-model" => {
                    tts_model = Some(PathBuf::from(value(&mut args, "--tts-model")?));
                }
                "--tts-voices" => {
                    tts_voices = Some(PathBuf::from(value(&mut args, "--tts-voices")?));
                }
                "--tts-tokens" => {
                    tts_tokens = Some(PathBuf::from(value(&mut args, "--tts-tokens")?));
                }
                "--tts-data-dir" => {
                    tts_data_dir = Some(PathBuf::from(value(&mut args, "--tts-data-dir")?));
                }
                "--speaker" => {
                    let raw = value(&mut args, "--speaker")?;
                    speaker = raw.parse().map_err(|_| {
                        format!("--speaker expects a non-negative integer, got {raw:?}")
                    })?;
                }
                "--speed" => {
                    let raw = value(&mut args, "--speed")?;
                    speed = raw
                        .parse()
                        .map_err(|_| format!("--speed expects a number, got {raw:?}"))?;
                }
                "--system-prompt" => {
                    system_prompt = Some(value(&mut args, "--system-prompt")?);
                }
                "--stt-threads" => {
                    let raw = value(&mut args, "--stt-threads")?;
                    stt_threads = raw.parse().map_err(|_| {
                        format!("--stt-threads expects a positive integer, got {raw:?}")
                    })?;
                    if stt_threads <= 0 {
                        return Err(format!(
                            "--stt-threads expects a positive integer, got {raw:?}"
                        ));
                    }
                }
                "--seconds" => {
                    let raw = value(&mut args, "--seconds")?;
                    seconds = Some(raw.parse().map_err(|_| {
                        format!("--seconds expects a non-negative integer, got {raw:?}")
                    })?);
                }
                other => {
                    return Err(format!(
                        "unknown argument {other:?} (expected --vad-model, --encoder, \
                         --merged-decoder, --tokens, --lm-model, --tts-model, --tts-voices, \
                         --tts-tokens, --tts-data-dir, --speaker, --speed, --system-prompt, \
                         --stt-threads, or --seconds)"
                    ));
                }
            }
        }

        Ok(Args {
            vad_model: required(vad_model, "--vad-model <silero_vad.onnx>")?,
            encoder: required(encoder, "--encoder <encoder_model.ort>")?,
            merged_decoder: required(
                merged_decoder,
                "--merged-decoder <decoder_model_merged.ort>",
            )?,
            tokens: required(tokens, "--tokens <tokens.txt>")?,
            lm_model: required(lm_model, "--lm-model <model.gguf>")?,
            tts_model: required(tts_model, "--tts-model <model.onnx>")?,
            tts_voices: required(tts_voices, "--tts-voices <voices.bin>")?,
            tts_tokens: required(tts_tokens, "--tts-tokens <tokens.txt>")?,
            tts_data_dir: required(tts_data_dir, "--tts-data-dir <espeak-ng-data>")?,
            speaker,
            speed,
            system_prompt: system_prompt.unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string()),
            stt_threads,
            seconds,
        })
    }

    fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
        value.ok_or_else(|| format!("{flag} is required"))
    }

    fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
        args.next().ok_or_else(|| format!("{flag} needs a value"))
    }
}
