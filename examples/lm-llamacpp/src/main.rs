//! Talk to a local language model: Sherpa VAD brackets each utterance from the
//! microphone, Moonshine v2 transcribes it, and a llama.cpp chat model streams
//! a reply back token by token.
//!
//! ```console
//! cargo run -p lm-llamacpp -- \
//!   --vad-model ./models/silero_vad.onnx \
//!   --encoder ./moonshine-model/encoder_model.ort \
//!   --merged-decoder ./moonshine-model/decoder_model_merged.ort \
//!   --tokens ./moonshine-model/tokens.txt \
//!   --lm-model ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
//! ```

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
fn main() {
    if let Err(error) = desktop::run() {
        eprintln!("lm-llamacpp: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("lm-llamacpp requires a desktop OS (macOS, Windows, or Linux)");
    std::process::exit(1);
}

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
mod desktop {
    use std::error::Error;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use futures::executor::block_on;
    use pipecrab::{
        DataFrame, Decision, Direction, Finality, Outbound, PipelineBuilder, Processor, Received,
        Role, Stage, StageError, SystemFrame, Transcript,
    };
    use pipecrab_audio::{AudioFormat, AudioSource, ResamplerStage};
    use pipecrab_audio_cpal::{CpalConfig, CpalSource};
    use pipecrab_lm::LmStage;
    use pipecrab_lm_llamacpp::{LlamaCpp, LlamaCppConfig};
    use pipecrab_stt::SttStage;
    use pipecrab_stt_sherpa::{MoonshineV2Config, OfflineSherpaStt};
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

    pub fn run() -> Result<(), Box<dyn Error>> {
        let Args {
            vad_model,
            encoder,
            merged_decoder,
            tokens,
            lm_model,
            system_prompt,
            stt_threads,
            seconds,
        } = parse_args()?;

        let audio_config = CpalConfig::default();
        let source = CpalSource::new(&audio_config)?;
        let mut vad_config = SherpaVadConfig::new(vad_model);
        vad_config.threshold = 0.35;
        vad_config.min_speech_duration = 0.1;
        vad_config.min_silence_duration = 0.5;
        vad_config.max_speech_duration = 30.0;
        let detector = SherpaVad::new(vad_config)?;
        let mut stt_config = MoonshineV2Config::new(encoder, merged_decoder, tokens);
        stt_config.num_threads = stt_threads;
        let transcriber = OfflineSherpaStt::new(stt_config)?;
        let resampler = ResamplerStage::new(SHERPA_FORMAT)?;

        println!("lm-llamacpp: loading {} …", lm_model.display());
        let model = LlamaCpp::load(LlamaCppConfig::new(lm_model))?;

        println!(
            "lm-llamacpp: input = {} @ {} Hz mono",
            source.device_name(),
            source.format().sample_rate
        );
        println!("lm-llamacpp: processing @ 16000 Hz mono");
        println!("lm-llamacpp: STT compute threads = {stt_threads}");
        println!(
            "lm-llamacpp: VAD threshold = 0.35, minimum speech = 100 ms, \
             pre-roll = 1000 ms, trailing silence = 500 ms"
        );
        match seconds {
            Some(seconds) => println!("lm-llamacpp: listening for {seconds} seconds"),
            None => println!("lm-llamacpp: listening until Ctrl-C"),
        }

        let max_chunks =
            seconds.map(|seconds| (seconds * 1000 / u64::from(audio_config.chunk_ms)) as usize);
        let (ends, driver) = PipelineBuilder::new()
            .stage(resampler)
            .stage(VadStage::with_config(
                detector,
                GateConfig {
                    preroll: Duration::from_secs(1),
                },
            ))
            .stage(SttStage::new(transcriber))
            .stage(UserTurnGate)
            .stage(LmStage::new(model, system_prompt))
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
                        eprintln!("lm-llamacpp: capture stopped: {error}");
                        break;
                    }
                }
            }
        };

        let drain = async move {
            let mut utterance_started = None;
            // Bytes of the in-flight agent reply already printed. Agent
            // partials are cumulative and append-only, so printing only the
            // unseen suffix streams the reply in place; `None` means no reply
            // is in flight.
            let mut reply_printed: Option<usize> = None;
            while let Some(received) = output.recv().await {
                match received {
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
                    Received::Data(DataFrame::Transcript(transcript))
                        if transcript.role == Role::Agent =>
                    {
                        let printed = match reply_printed {
                            Some(printed) => printed,
                            None => {
                                print!("Agent: ");
                                0
                            }
                        };
                        print!("{}", transcript.text.get(printed..).unwrap_or_default());
                        let _ = std::io::stdout().flush();
                        match transcript.finality {
                            Finality::Partial { .. } => {
                                reply_printed = Some(transcript.text.len());
                            }
                            Finality::Final => {
                                println!();
                                reply_printed = None;
                            }
                        }
                    }
                    Received::Sys(_, SystemFrame::Error { message, fatal: _ }) => {
                        eprintln!("lm-llamacpp: pipeline error: {message}")
                    }
                    Received::Data(_) | Received::Sys(_, _) => {}
                }
            }
        };

        block_on(async { futures::join!(driver, pump_in, drain) });
        println!("lm-llamacpp: stopped");
        Ok(())
    }

    struct Args {
        vad_model: PathBuf,
        encoder: PathBuf,
        merged_decoder: PathBuf,
        tokens: PathBuf,
        lm_model: PathBuf,
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
                         --merged-decoder, --tokens, --lm-model, --system-prompt, \
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
