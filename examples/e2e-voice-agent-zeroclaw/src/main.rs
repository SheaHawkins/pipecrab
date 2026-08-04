//! The local voice pipeline with a ZeroClaw daemon as the brain.
//!
//! The pipeline is an RPC peer of a running `zeroclaw daemon`: each utterance
//! becomes a `session/prompt`, streamed reply chunks become speech, and the
//! conversation is a first-class daemon session — open the ZeroClaw TUI
//! against the same daemon and watch the transcript update turn by turn.
//!
//! ```text
//!   CpalSource (mic)
//!       ▼
//!   ResamplerStage (16 kHz mono)
//!       ▼
//!   VadStage<SherpaVad> ──▶ SttStage<OfflineSherpaStt>
//!       ▼
//!   BargeInStage           (speech onset ⇒ downstream Interrupt)
//!       ▼
//!   UserTurnGate
//!       ▼
//!   DispatchIngress<ZeroclawDelegateSource>  (delegation results re-enter here)
//!       ▼
//!   LmStage<ZeroclawLm>    (no tools: the daemon's agent loop owns them)
//!       ▼
//!   ToolCallEcho ──▶ DispatchEcho
//!       ▼
//!   SentenceChunker ──▶ AgentEcho ──▶ TtsStage<SherpaTts>
//!       ▼
//!   ResamplerStage (device rate) ──▶ CpalSink (speaker)
//! ```
//!
//! There is no dispatch egress and no dispatch sink: tool calls never leave
//! the pipeline. The daemon executes tools inline during the turn, and long
//! work goes through ZeroClaw's `delegate` tool with `background: true`. The
//! `ZeroclawDelegateSource` watches the session workspace's
//! `delegate_results/` and a finished task re-enters through ingress as a
//! `[dispatch/completion]` turn the agent speaks.
//!
//! The agent profile does the heavy lifting: it must use a provider that
//! streams with tool events (OpenRouter, Anthropic, OpenAI, or an
//! OpenAI-compatible local server with native tool calling — not native
//! `ollama`), keep the tool registry trimmed for voice, and allow `delegate`
//! in background mode. The system prompt lives there too.
//!
//! **Use headphones** — over speakers the microphone re-captures the agent's
//! own voice and it talks to itself.

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn main() {
    if let Err(error) = desktop::run() {
        eprintln!("e2e-voice-agent-zeroclaw: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn main() {
    eprintln!(
        "e2e-voice-agent-zeroclaw requires an OS with unix domain sockets \
         (macOS or Linux) and a local zeroclaw daemon"
    );
    std::process::exit(1);
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
mod desktop {
    use std::collections::VecDeque;
    use std::error::Error;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use pipecrab::{
        DataFrame, Decision, Direction, DispatchEvent, DispatchFrame, Finality, ModelFrame,
        Outbound, PipelineBuilder, Processor, Received, Role, Stage, StageError, SystemFrame,
        Transcript,
    };
    use pipecrab_audio::{AudioFormat, AudioSink, AudioSource, ResamplerStage};
    use pipecrab_audio_cpal::{CpalConfig, CpalSink, CpalSource};
    use pipecrab_dispatch::DispatchIngress;
    use pipecrab_lm::LmStage;
    use pipecrab_lm_zeroclaw::{ZeroclawLmConfig, connect};
    use pipecrab_stt::SttStage;
    use pipecrab_stt_sherpa::{MoonshineV2Config, OfflineSherpaStt};
    use pipecrab_tts::{SentenceChunker, Synthesizer, TtsStage};
    use pipecrab_tts_sherpa::{KokoroConfig, SherpaTts};
    use pipecrab_turn::BargeInStage;
    use pipecrab_vad::{GateConfig, VadStage};
    use pipecrab_vad_sherpa::{SherpaVad, SherpaVadConfig};

    const SHERPA_FORMAT: AudioFormat = AudioFormat {
        sample_rate: 16_000,
        channels: 1,
    };

    /// The stage conversation is dead weight behind this adapter — the daemon
    /// session owns the real history and the real system prompt.
    const STUB_SYSTEM_PROMPT: &str =
        "(unused: the system prompt lives in the ZeroClaw agent profile)";

    /// Prints each completed user utterance and drops empty finals so a noise
    /// trigger never becomes a daemon turn (which rejects blank prompts).
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
                    true => Decision::drop(),
                    false => Decision::forward().emit(PrintUser(text.clone())),
                },
                _ => Decision::forward(),
            }
        }
    }

    #[async_trait]
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

    /// Prints the daemon's tool calls passing by — observability only; the
    /// daemon already executed them by the time they print.
    struct ToolCallEcho;

    /// One tool call to print.
    struct PrintToolCall(String);

    impl Processor for ToolCallEcho {
        type Effect = PrintToolCall;

        fn decide_data(&mut self, frame: &DataFrame) -> Decision<PrintToolCall> {
            match frame {
                DataFrame::Model(ModelFrame::ToolCall(call)) => Decision::forward().emit(
                    PrintToolCall(format!("{} {}", call.name, call.arguments_json)),
                ),
                _ => Decision::forward(),
            }
        }
    }

    #[async_trait]
    impl Stage for ToolCallEcho {
        async fn perform(
            &self,
            PrintToolCall(line): PrintToolCall,
            _out: &Outbound,
        ) -> Result<(), StageError> {
            println!("[tool] {line}");
            Ok(())
        }
    }

    /// Prints delegation lifecycle events injected by ingress.
    struct DispatchEcho;

    /// One line describing a dispatch frame.
    struct PrintDispatch(String);

    impl Processor for DispatchEcho {
        type Effect = PrintDispatch;

        fn decide_data(&mut self, frame: &DataFrame) -> Decision<PrintDispatch> {
            let DataFrame::Dispatch(DispatchFrame::Event(event)) = frame else {
                return Decision::forward();
            };
            let line = match event {
                DispatchEvent::Completion { message, .. } => format!("completed: {message}"),
                DispatchEvent::Failure {
                    message, retryable, ..
                } => format!("failed (retryable={retryable}): {message}"),
                DispatchEvent::Accepted { task_id, .. } => format!("accepted, task {task_id}"),
                DispatchEvent::Rejected { message, .. } => format!("rejected: {message}"),
                DispatchEvent::Progress { message, .. } => format!("progress: {message}"),
                DispatchEvent::Question { message, .. } => format!("question: {message}"),
            };
            Decision::forward().emit(PrintDispatch(line))
        }
    }

    #[async_trait]
    impl Stage for DispatchEcho {
        async fn perform(
            &self,
            PrintDispatch(line): PrintDispatch,
            _out: &Outbound,
        ) -> Result<(), StageError> {
            println!("[task] {line}");
            Ok(())
        }
    }

    /// Prints each completed agent sentence on its way to synthesis.
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

    #[async_trait]
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
            tts_model,
            tts_voices,
            tts_tokens,
            tts_data_dir,
            speaker,
            speed,
            stt_threads,
            seconds,
            agent,
            socket,
            session_id,
        } = parse_args()?;

        let audio_config = CpalConfig::default();
        let source = CpalSource::new(&audio_config)?;
        let sink = CpalSink::new(&audio_config)?;
        let mut vad_config = SherpaVadConfig::new(vad_model);
        vad_config.threshold = 0.35;
        // High enough that a cough or bump does not barge in and kill a reply.
        vad_config.min_speech_duration = 0.25;
        vad_config.min_silence_duration = 0.5;
        vad_config.max_speech_duration = 30.0;
        let detector = SherpaVad::new(vad_config)?;
        let mut stt_config = MoonshineV2Config::new(encoder, merged_decoder, tokens);
        stt_config.num_threads = stt_threads;
        let transcriber = OfflineSherpaStt::new(stt_config)?;
        let capture_resampler = ResamplerStage::new(SHERPA_FORMAT)?;
        let playback_resampler = ResamplerStage::new(sink.format())?;

        let mut tts_config = KokoroConfig::new(tts_model, tts_voices, tts_tokens, tts_data_dir);
        tts_config.speaker = speaker;
        tts_config.speed = speed;
        let synth = SherpaTts::new(tts_config)?;

        println!(
            "e2e-voice-agent-zeroclaw: input = {} @ {} Hz mono",
            source.device_name(),
            source.format().sample_rate
        );
        println!(
            "e2e-voice-agent-zeroclaw: output = {} @ {} Hz",
            sink.device_name(),
            sink.format().sample_rate
        );
        println!(
            "e2e-voice-agent-zeroclaw: kokoro @ {} Hz mono, speaker {speaker}, speed {speed}",
            synth.output_format().sample_rate
        );
        println!("e2e-voice-agent-zeroclaw: use headphones so the agent does not hear itself");
        match seconds {
            Some(seconds) => println!("e2e-voice-agent-zeroclaw: listening for {seconds} seconds"),
            None => println!("e2e-voice-agent-zeroclaw: listening until Ctrl-C"),
        }

        let ring_ms = u64::from(audio_config.chunk_ms) * audio_config.ring_chunks as u64;
        let max_chunks =
            seconds.map(|seconds| (seconds * 1000 / u64::from(audio_config.chunk_ms)) as usize);

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        runtime.block_on(async move {
            let mut lm_config = ZeroclawLmConfig::new(agent.as_str());
            if let Some(socket) = socket {
                lm_config = lm_config.with_socket_path(socket);
            }
            if let Some(session_id) = &session_id {
                lm_config = lm_config.with_session_id(session_id.as_str());
            }
            let (model, delegate_source) = connect(lm_config).await?;
            println!(
                "e2e-voice-agent-zeroclaw: agent {agent:?}, session {} — open the ZeroClaw TUI \
                 to watch this conversation",
                model.session_id()
            );
            println!(
                "e2e-voice-agent-zeroclaw: watching {} for background delegations",
                model.workspace_dir().join("delegate_results").display()
            );

            let lm = LmStage::new(model, STUB_SYSTEM_PROMPT);

            let (ends, driver) = PipelineBuilder::new()
                .stage(capture_resampler)
                .stage(VadStage::with_config(
                    detector,
                    GateConfig {
                        preroll: Duration::from_secs(1),
                    },
                ))
                .stage(SttStage::new(transcriber))
                .stage(BargeInStage::new())
                .stage(UserTurnGate)
                .stage(DispatchIngress::new(delegate_source))
                .stage(lm)
                .stage(ToolCallEcho)
                .stage(DispatchEcho)
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
                            eprintln!("e2e-voice-agent-zeroclaw: capture stopped: {error}");
                            break;
                        }
                    }
                }
            };

            let pump_out = async move {
                let mut sink = sink;
                let mut utterance_started = None;
                let mut awaiting_reply: Option<Instant> = None;
                // Keepers from the tail-lane flush on a barge-in, replayed
                // ahead of the next receive.
                let mut pending: VecDeque<DataFrame> = VecDeque::new();
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
                            if let Some(asked) = awaiting_reply.take() {
                                println!(
                                    "  ⏱ first speech {:.2} s after silence",
                                    asked.elapsed().as_secs_f32()
                                );
                            }
                            if let Err(error) = sink.play(chunk).await {
                                eprintln!("e2e-voice-agent-zeroclaw: playback stopped: {error}");
                                break;
                            }
                        }
                        Received::Data(DataFrame::SpeechStarted) => {
                            utterance_started = Some(Instant::now());
                            awaiting_reply = None;
                            println!("SpeechStarted");
                        }
                        Received::Data(DataFrame::SpeechStopped) => {
                            let elapsed = utterance_started.take().map(|s| s.elapsed());
                            awaiting_reply = Some(Instant::now());
                            match elapsed {
                                Some(elapsed) => {
                                    println!("SpeechStopped ({:.2} s)", elapsed.as_secs_f32());
                                }
                                None => println!("SpeechStopped"),
                            }
                        }
                        Received::Sys(_, SystemFrame::Interrupt) => {
                            // Barge-in: drop what the device already holds,
                            // then the agent audio still queued past the tail —
                            // nothing flushes the tail lane for the app.
                            sink.cancel();
                            pending.extend(output.flush_data());
                        }
                        Received::Sys(_, SystemFrame::Error { message, fatal: _ }) => {
                            eprintln!("e2e-voice-agent-zeroclaw: pipeline error: {message}")
                        }
                        Received::Data(_) | Received::Sys(_, _) => {}
                    }
                }
                std::thread::sleep(Duration::from_millis(ring_ms + 50));
            };

            futures::join!(driver, pump_in, pump_out);
            Ok::<(), Box<dyn Error>>(())
        })?;

        println!("e2e-voice-agent-zeroclaw: stopped");
        Ok(())
    }

    struct Args {
        vad_model: PathBuf,
        encoder: PathBuf,
        merged_decoder: PathBuf,
        tokens: PathBuf,
        tts_model: PathBuf,
        tts_voices: PathBuf,
        tts_tokens: PathBuf,
        tts_data_dir: PathBuf,
        speaker: i32,
        speed: f32,
        stt_threads: i32,
        seconds: Option<u64>,
        agent: String,
        socket: Option<PathBuf>,
        session_id: Option<String>,
    }

    fn parse_args() -> Result<Args, String> {
        let mut vad_model = None;
        let mut encoder = None;
        let mut merged_decoder = None;
        let mut tokens = None;
        let mut tts_model = None;
        let mut tts_voices = None;
        let mut tts_tokens = None;
        let mut tts_data_dir = None;
        let mut speaker = 0;
        let mut speed = 1.0;
        let mut stt_threads = 2;
        let mut seconds = None;
        let mut agent = None;
        let mut socket = None;
        let mut session_id = None;
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
                "--agent" => agent = Some(value(&mut args, "--agent")?),
                "--zeroclaw-socket" => {
                    socket = Some(PathBuf::from(value(&mut args, "--zeroclaw-socket")?));
                }
                "--session-id" => session_id = Some(value(&mut args, "--session-id")?),
                other => {
                    return Err(format!(
                        "unknown argument {other:?} (expected --vad-model, --encoder, \
                         --merged-decoder, --tokens, --tts-model, --tts-voices, --tts-tokens, \
                         --tts-data-dir, --speaker, --speed, --stt-threads, --seconds, --agent, \
                         --zeroclaw-socket, or --session-id)"
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
            tts_model: required(tts_model, "--tts-model <model.onnx>")?,
            tts_voices: required(tts_voices, "--tts-voices <voices.bin>")?,
            tts_tokens: required(tts_tokens, "--tts-tokens <tokens.txt>")?,
            tts_data_dir: required(tts_data_dir, "--tts-data-dir <espeak-ng-data>")?,
            speaker,
            speed,
            stt_threads,
            seconds,
            agent: required(agent, "--agent <zeroclaw agent alias>")?,
            socket,
            session_id,
        })
    }

    fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
        value.ok_or_else(|| format!("{flag} is required"))
    }

    fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
        args.next().ok_or_else(|| format!("{flag} needs a value"))
    }
}
