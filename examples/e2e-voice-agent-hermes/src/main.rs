//! The local voice agent with a Hermes Agent dispatch transport wired in: a
//! spoken errand becomes a background Hermes run, and the agent speaks the
//! result whenever it lands — minutes later, mid-conversation, without blocking
//! the turn it was asked in.
//!
//! ```text
//!   CpalSource (mic)
//!       ▼
//!   ResamplerStage (16 kHz mono)
//!       ▼
//!   VadStage<SherpaVad> ──▶ SttStage<OfflineSherpaStt>
//!       ▼
//!   BargeInStage               (speech onset ⇒ downstream Interrupt)
//!       ▼
//!   UserTurnGate
//!       ▼
//!   DispatchIngress            (injects Hermes events into the idle pipeline)
//!       ▼
//!   LmStage<LlamaCpp>          (dispatch tools configured; speaks completions)
//!       ▼
//!   DispatchEgress ──▶ HermesSink ──▶ POST /v1/runs
//!       ▼
//!   DispatchAck                (speaks the acknowledgement the model does not)
//!       ▼
//!   DispatchEcho               (prints the task lifecycle)
//!       ▼
//!   SentenceChunker ──▶ AgentEcho ──▶ TtsStage<SherpaTts>
//!       ▼
//!   ResamplerStage (device rate) ──▶ CpalSink (speaker)
//! ```
//!
//! Ingress sits above the model so an event that arrives while nothing is
//! flowing still reaches it; egress sits below so the model's tool calls leave
//! through the transport. The Hermes poller runs on its own tokio task, so a
//! task completing during silence still wakes the pipeline.
//!
//! The local model decides when to dispatch: `LmStage::with_tools` hands it
//! `dispatch_task`, and `pipecrab-lm-llamacpp` renders it into the prompt and
//! reads calls back out. That needs a GGUF the adapter's default `ChatMlXml`
//! dialect covers — Qwen 2.5/3 or Nous Hermes.
//!
//! Qwen 3 reasons before it answers, and this pipeline speaks every token that
//! is not tool-call syntax, so the default suppresses it by prefilling the
//! assistant turn with Qwen 3's own empty think block. `--thinking` leaves it
//! on, which is what a Qwen 2.5 GGUF wants — it never saw those tokens.
//!
//! `update_task` is deliberately out of scope here; see the comment in `run`.
//!
//! **Use headphones** — over speakers the microphone re-captures the agent's
//! own voice and it talks to itself.

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
fn main() {
    if let Err(error) = desktop::run() {
        eprintln!("e2e-voice-agent-hermes: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("e2e-voice-agent-hermes requires a desktop OS (macOS, Windows, or Linux)");
    std::process::exit(1);
}

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
mod desktop {
    use std::collections::VecDeque;
    use std::error::Error;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use futures::StreamExt;
    use pipecrab::{
        DataFrame, Decision, Direction, DispatchCommand, DispatchEvent, DispatchFrame, Finality,
        Outbound, PipelineBuilder, Processor, Received, Role, Stage, StageError, SystemFrame,
        Transcript,
    };
    use pipecrab_audio::{AudioFormat, AudioSink, AudioSource, ResamplerStage};
    use pipecrab_audio_cpal::{CpalConfig, CpalSink, CpalSource};
    use pipecrab_dispatch::{Dispatch, dispatch_task_definition};
    use pipecrab_dispatch_hermes::{HermesConfig, connect};
    use pipecrab_lm::{Conversation, GenParams, LanguageModel, LmStage, Message, ToolDefinition};
    use pipecrab_lm_llamacpp::{LlamaCpp, LlamaCppConfig};
    use pipecrab_stt::SttStage;
    use pipecrab_stt_sherpa::{MoonshineV2Config, OfflineSherpaStt};
    use pipecrab_tts::{SentenceChunker, Synthesizer, TtsStage};
    use pipecrab_tts_sherpa::{KokoroConfig, SherpaTts};
    use pipecrab_turn::BargeInStage;
    use pipecrab_vad::{GateConfig, VadStage};
    use pipecrab_vad_sherpa::{SherpaVad, SherpaVadConfig};
    use url::Url;

    const SHERPA_FORMAT: AudioFormat = AudioFormat {
        sample_rate: 16_000,
        channels: 1,
    };

    const DEFAULT_HERMES_URL: &str = "http://127.0.0.1:8642";

    /// Spoken when the errand does not read as a command — see [`acknowledge`].
    const GENERIC_ACK: &str = "Alright, I'm on it.";

    /// Qwen 3's non-thinking mode: what its `enable_thinking=false` branch
    /// emits. Reasoning read aloud is worse than useless in a spoken agent, and
    /// at `gpu_layers: 0` the tokens are latency the user waits through.
    const QWEN3_NO_THINK: &str = "<think>\n\n</think>\n\n";

    const DEFAULT_SYSTEM_PROMPT: &str = "You are a friendly voice assistant. \
         Replies are spoken aloud, so answer in one or two short sentences of \
         plain prose with no markup. Answer from your own knowledge whatever \
         you can. Anything else — the web, the world outside this machine, \
         work that outlives this turn — is an errand: call `dispatch_task`. \
         Write `task` as a short command naming the errand, the way you would \
         finish the sentence \"Let me …\" — `check the weather in Denver`, \
         `find a crab recipe`, `look up flights to Tokyo`. Never a question, \
         never a full sentence, and never ask the user to repeat or narrow the \
         errand. When one reports back, relay its answer in the same short \
         spoken style.";

    /// Prints each completed user utterance and drops empty finals so a noise
    /// trigger never wakes the language model; every other frame forwards
    /// untouched.
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

    /// Speaks one line when an errand leaves, because the model does not.
    ///
    /// A tool-calling turn is the call and nothing else — Qwen emits
    /// `<tool_call>` immediately, with no text beside it — so a dispatch would
    /// otherwise be silent until the task reports back, and the user would hear
    /// nothing at all in reply. This is the audible form of the
    /// `[task] dispatching:` line [`DispatchEcho`] prints: app feedback, not the
    /// model's voice.
    struct DispatchAck;

    /// One errand's acknowledgement, ready to speak.
    struct SpeakAck(Arc<str>);

    impl Processor for DispatchAck {
        type Effect = SpeakAck;

        fn decide_data(&mut self, frame: &DataFrame) -> Decision<SpeakAck> {
            match frame {
                DataFrame::Dispatch(DispatchFrame::Command(DispatchCommand::Create {
                    task,
                    ..
                })) => Decision::forward().emit(SpeakAck(acknowledge(task))),
                _ => Decision::forward(),
            }
        }
    }

    #[async_trait]
    impl Stage for DispatchAck {
        async fn perform(
            &self,
            SpeakAck(text): SpeakAck,
            out: &Outbound,
        ) -> Result<(), StageError> {
            let _ = out
                .send_data(DataFrame::Transcript(Transcript {
                    role: Role::Agent,
                    finality: Finality::Final,
                    text,
                }))
                .await;
            Ok(())
        }
    }

    /// Decode one token so the first real turn does not pay for the prompt.
    ///
    /// The system message and the tool declarations are a constant prefix of
    /// every turn, and the worker reuses the longest prefix it has already
    /// decoded — so processing them once here takes several hundred tokens of
    /// prefill off the first thing the user says. Cold, that is the difference
    /// between answering and appearing to have missed the question.
    ///
    /// Failure is not fatal: a warm-up that errors leaves a cold cache, which is
    /// exactly where we started.
    async fn warm_up(model: &LlamaCpp, system_prompt: &str, tools: &[ToolDefinition]) {
        print!("e2e-voice-agent-hermes: warming the model … ");
        let _ = std::io::stdout().flush();
        let started = Instant::now();
        let conversation = Conversation {
            messages: vec![
                Message::system(system_prompt.to_string()),
                Message::user("Hello."),
            ],
        };
        let params = GenParams {
            max_tokens: Some(1),
            ..GenParams::default()
        };
        match model.generate(&conversation, &params, tools).await {
            Ok(stream) => {
                let _ = stream.collect::<Vec<_>>().await;
                println!("{:.1} s", started.elapsed().as_secs_f32());
            }
            Err(error) => println!("skipped ({error})"),
        }
    }

    /// `"Let me <task>."`, naming the errand back to the user.
    ///
    /// The system prompt asks for `task` as the phrase completing "Let me …",
    /// which is what makes this read. A model that writes a question there
    /// instead gets [`GENERIC_ACK`]: "Let me what's the weather in Denver?" is
    /// worse than saying nothing specific.
    fn acknowledge(task: &str) -> Arc<str> {
        let task = task.trim().trim_end_matches(['.', '!']);
        match completes_let_me(task) {
            true => Arc::from(format!("Let me {task}.")),
            false => Arc::from(GENERIC_ACK),
        }
    }

    /// Whether `task` reads as a command rather than a question.
    fn completes_let_me(task: &str) -> bool {
        if task.is_empty() || task.ends_with('?') {
            return false;
        }
        let first = task
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .trim_end_matches([',', ':'])
            .to_lowercase();
        !matches!(
            first.as_str(),
            "what"
                | "what's"
                | "whats"
                | "when"
                | "where"
                | "who"
                | "why"
                | "how"
                | "which"
                | "is"
                | "are"
                | "was"
                | "were"
                | "can"
                | "could"
                | "would"
                | "should"
                | "will"
                | "do"
                | "does"
                | "did"
                | "i"
                | "you"
                | "it"
                | "there"
        )
    }

    /// Prints the dispatch traffic passing by — the commands egress published
    /// and the events ingress injected — so the task lifecycle is visible.
    struct DispatchEcho;

    /// One line describing a dispatch frame.
    struct PrintDispatch(String);

    impl Processor for DispatchEcho {
        type Effect = PrintDispatch;

        fn decide_data(&mut self, frame: &DataFrame) -> Decision<PrintDispatch> {
            let DataFrame::Dispatch(dispatch) = frame else {
                return Decision::forward();
            };
            let line = match dispatch {
                DispatchFrame::Event(DispatchEvent::Accepted { task_id, .. }) => {
                    format!("accepted, task {task_id}")
                }
                DispatchFrame::Event(DispatchEvent::Rejected { message, .. }) => {
                    format!("rejected: {message}")
                }
                DispatchFrame::Event(DispatchEvent::Progress { message, .. }) => {
                    format!("progress: {message}")
                }
                DispatchFrame::Event(DispatchEvent::Question { message, .. }) => {
                    format!("question: {message}")
                }
                DispatchFrame::Event(DispatchEvent::Completion { message, .. }) => {
                    format!("completed: {message}")
                }
                DispatchFrame::Event(DispatchEvent::Failure {
                    message, retryable, ..
                }) => format!("failed (retryable={retryable}): {message}"),
                // Egress echoes each command downstream after publishing it.
                DispatchFrame::Command(DispatchCommand::Create { task, .. }) => {
                    format!("dispatching: {task}")
                }
                DispatchFrame::Command(DispatchCommand::Update {
                    task_id, message, ..
                }) => format!("updating {task_id}: {message}"),
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
            hermes_url,
            hermes_key,
            thinking,
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

        println!("e2e-voice-agent-hermes: loading {} …", lm_model.display());
        let mut lm_config = LlamaCppConfig::new(lm_model);
        if !thinking {
            lm_config = lm_config.with_assistant_prefix(QWEN3_NO_THINK);
        }
        let model = LlamaCpp::load(lm_config)?;

        let mut tts_config = KokoroConfig::new(tts_model, tts_voices, tts_tokens, tts_data_dir);
        tts_config.speaker = speaker;
        tts_config.speed = speed;
        let synth = SherpaTts::new(tts_config)?;

        println!(
            "e2e-voice-agent-hermes: input = {} @ {} Hz mono",
            source.device_name(),
            source.format().sample_rate
        );
        println!(
            "e2e-voice-agent-hermes: output = {} @ {} Hz",
            sink.device_name(),
            sink.format().sample_rate
        );
        println!(
            "e2e-voice-agent-hermes: kokoro @ {} Hz mono, speaker {speaker}, speed {speed}",
            synth.output_format().sample_rate
        );
        println!(
            "e2e-voice-agent-hermes: thinking = {}",
            match thinking {
                true => "on (the model's reasoning is spoken)",
                false => "suppressed",
            }
        );
        println!("e2e-voice-agent-hermes: hermes = {hermes_url}");
        println!("e2e-voice-agent-hermes: ask for an errand and the model dispatches it");
        println!("e2e-voice-agent-hermes: use headphones so the agent does not hear itself");
        match seconds {
            Some(seconds) => println!("e2e-voice-agent-hermes: listening for {seconds} seconds"),
            None => println!("e2e-voice-agent-hermes: listening until Ctrl-C"),
        }

        let ring_ms = u64::from(audio_config.chunk_ms) * audio_config.ring_chunks as u64;
        let max_chunks =
            seconds.map(|seconds| (seconds * 1000 / u64::from(audio_config.chunk_ms)) as usize);

        // `connect` spawns the poller, so everything runs inside this runtime.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        runtime.block_on(async move {
            let hermes = HermesConfig::new(hermes_key).with_base_url(hermes_url);
            let (dispatch_source, dispatch_sink) = connect(hermes);
            let dispatch = Dispatch::new(dispatch_source, dispatch_sink);
            let (ingress, egress) = dispatch.into_stages();

            warm_up(&model, &system_prompt, &[dispatch_task_definition()]).await;

            // Only `dispatch_task`, not the full `dispatch_tool_definitions()`:
            // a 0.5B model cannot hold two similar tools apart, and reaches for
            // `update_task` with an invented `task_id` instead of starting a
            // task. Egress translates both regardless, so a model with room for
            // the distinction gets the pair back by widening this one line.
            let lm = LmStage::with_tools(model, system_prompt, [dispatch_task_definition()])?;

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
                .stage(ingress)
                .stage(lm)
                .stage(egress)
                .stage(DispatchAck)
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
                            eprintln!("e2e-voice-agent-hermes: capture stopped: {error}");
                            break;
                        }
                    }
                }
            };

            let pump_out = async move {
                let mut sink = sink;
                let mut utterance_started = None;
                // Set when the user stops speaking, cleared by the first audio
                // that answers them: STT drops every capture chunk, so the only
                // audio reaching this end is synthesis.
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
                                eprintln!("e2e-voice-agent-hermes: playback stopped: {error}");
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
                            eprintln!("e2e-voice-agent-hermes: pipeline error: {message}")
                        }
                        Received::Data(_) | Received::Sys(_, _) => {}
                    }
                }
                // The device is still draining its ring; dropping the sink now
                // would clip the agent's last words.
                std::thread::sleep(Duration::from_millis(ring_ms + 50));
            };

            futures::join!(driver, pump_in, pump_out);
            Ok::<(), Box<dyn Error>>(())
        })?;

        println!("e2e-voice-agent-hermes: stopped");
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
        hermes_url: Url,
        hermes_key: String,
        thinking: bool,
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
        let mut hermes_url = None;
        let mut hermes_key = None;
        let mut thinking = false;
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
                "--hermes-url" => {
                    let raw = value(&mut args, "--hermes-url")?;
                    hermes_url = Some(
                        Url::parse(&raw)
                            .map_err(|e| format!("--hermes-url {raw:?} is not a URL: {e}"))?,
                    );
                }
                "--hermes-key" => hermes_key = Some(value(&mut args, "--hermes-key")?),
                "--thinking" => thinking = true,
                other => {
                    return Err(format!(
                        "unknown argument {other:?} (expected --vad-model, --encoder, \
                         --merged-decoder, --tokens, --lm-model, --tts-model, --tts-voices, \
                         --tts-tokens, --tts-data-dir, --speaker, --speed, --system-prompt, \
                         --stt-threads, --seconds, --hermes-url, --hermes-key, or --thinking)"
                    ));
                }
            }
        }

        // The key is accepted on the command line or, to keep it out of shell
        // history, from the environment.
        let hermes_key = hermes_key
            .or_else(|| std::env::var("HERMES_API_KEY").ok())
            .ok_or("--hermes-key is required (or set HERMES_API_KEY)")?;

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
            hermes_url: hermes_url.unwrap_or_else(|| {
                Url::parse(DEFAULT_HERMES_URL).expect("the default URL is valid")
            }),
            hermes_key,
            thinking,
        })
    }

    fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
        value.ok_or_else(|| format!("{flag} is required"))
    }

    fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
        args.next().ok_or_else(|| format!("{flag} needs a value"))
    }

    #[cfg(test)]
    mod tests {
        use super::{GENERIC_ACK, acknowledge};

        #[test]
        fn a_command_is_spoken_back() {
            // What the prompt asks the model for, and what it produces.
            for task in [
                "check the weather in Denver",
                "find a crab recipe",
                "look up flights to Tokyo next month",
            ] {
                assert_eq!(&*acknowledge(task), format!("Let me {task}."));
            }
        }

        #[test]
        fn a_question_falls_back_instead_of_reading_as_nonsense() {
            for task in [
                "What's the weather like in Denver today?",
                "How much is a flight to Tokyo?",
                "is it raining",
            ] {
                assert_eq!(&*acknowledge(task), GENERIC_ACK, "task: {task:?}");
            }
        }

        #[test]
        fn punctuation_is_not_doubled() {
            assert_eq!(
                &*acknowledge("find a crab recipe."),
                "Let me find a crab recipe."
            );
            assert_eq!(
                &*acknowledge("  find a crab recipe  "),
                "Let me find a crab recipe."
            );
        }
    }
}
