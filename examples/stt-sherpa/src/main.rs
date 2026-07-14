//! Transcribe speech from the default microphone with Sherpa VAD and streaming
//! STT.
//!
//! ```console
//! cargo run -p stt-sherpa -- \
//!   --vad-model ./silero_vad.onnx \
//!   --encoder ./streaming-model/encoder-epoch-99-avg-1.int8.onnx \
//!   --decoder ./streaming-model/decoder-epoch-99-avg-1.onnx \
//!   --joiner ./streaming-model/joiner-epoch-99-avg-1.int8.onnx \
//!   --tokens ./streaming-model/tokens.txt
//! ```

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
fn main() {
    if let Err(error) = desktop::run() {
        eprintln!("stt-sherpa: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("stt-sherpa requires a desktop OS (macOS, Windows, or Linux)");
    std::process::exit(1);
}

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
mod desktop {
    use std::error::Error;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use futures::executor::block_on;
    use pipecrab::{DataFrame, Direction, Finality, PipelineBuilder, Received, SystemFrame};
    use pipecrab_audio::{AudioFormat, AudioSource, ResamplerStage};
    use pipecrab_audio_cpal::{CpalConfig, CpalSource};
    use pipecrab_stt::SttStage;
    use pipecrab_stt_sherpa::{SherpaStt, SherpaSttConfig};
    use pipecrab_vad::{GateConfig, VadStage};
    use pipecrab_vad_sherpa::{SherpaVad, SherpaVadConfig};

    const SHERPA_FORMAT: AudioFormat = AudioFormat {
        sample_rate: 16_000,
        channels: 1,
    };

    pub fn run() -> Result<(), Box<dyn Error>> {
        let Args {
            vad_model,
            encoder,
            decoder,
            joiner,
            tokens,
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
        let mut stt_config = SherpaSttConfig::new(encoder, decoder, joiner, tokens);
        stt_config.num_threads = stt_threads;
        let transcriber = SherpaStt::new(stt_config)?;
        let resampler = ResamplerStage::new(SHERPA_FORMAT)?;

        println!(
            "stt-sherpa: input = {} @ {} Hz mono",
            source.device_name(),
            source.format().sample_rate
        );
        println!("stt-sherpa: processing @ 16000 Hz mono");
        println!("stt-sherpa: STT compute threads = {stt_threads}");
        println!(
            "stt-sherpa: VAD threshold = 0.35, minimum speech = 100 ms, \
             pre-roll = 1000 ms, trailing silence = 500 ms"
        );
        match seconds {
            Some(seconds) => println!("stt-sherpa: listening for {seconds} seconds"),
            None => println!("stt-sherpa: listening until Ctrl-C"),
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
                        eprintln!("stt-sherpa: capture stopped: {error}");
                        break;
                    }
                }
            }
        };

        let drain = async move {
            let mut utterance_started = None;
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
                    Received::Data(DataFrame::Transcript(transcript)) => {
                        match transcript.finality {
                            Finality::Partial { .. } => println!("Partial: {}", transcript.text),
                            Finality::Final if transcript.text.is_empty() => {
                                println!("Final: <no speech recognized>");
                            }
                            Finality::Final => println!("Final: {}", transcript.text),
                        }
                    }
                    Received::Sys(_, SystemFrame::Error { message, fatal: _ }) => {
                        eprintln!("stt-sherpa: pipeline error: {message}")
                    }
                    Received::Data(_) | Received::Sys(_, _) => {}
                }
            }
        };

        block_on(async { futures::join!(driver, pump_in, drain) });
        println!("stt-sherpa: stopped");
        Ok(())
    }

    struct Args {
        vad_model: PathBuf,
        encoder: PathBuf,
        decoder: PathBuf,
        joiner: PathBuf,
        tokens: PathBuf,
        stt_threads: i32,
        seconds: Option<u64>,
    }

    fn parse_args() -> Result<Args, String> {
        let mut vad_model = None;
        let mut encoder = None;
        let mut decoder = None;
        let mut joiner = None;
        let mut tokens = None;
        let mut stt_threads = 2;
        let mut seconds = None;
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--vad-model" => {
                    vad_model = Some(PathBuf::from(value(&mut args, "--vad-model")?));
                }
                "--encoder" => encoder = Some(PathBuf::from(value(&mut args, "--encoder")?)),
                "--decoder" => decoder = Some(PathBuf::from(value(&mut args, "--decoder")?)),
                "--joiner" => joiner = Some(PathBuf::from(value(&mut args, "--joiner")?)),
                "--tokens" => tokens = Some(PathBuf::from(value(&mut args, "--tokens")?)),
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
                         --decoder, --joiner, --tokens, --stt-threads, or --seconds)"
                    ));
                }
            }
        }

        Ok(Args {
            vad_model: required(vad_model, "--vad-model <silero_vad.onnx>")?,
            encoder: required(encoder, "--encoder <encoder.onnx>")?,
            decoder: required(decoder, "--decoder <decoder.onnx>")?,
            joiner: required(joiner, "--joiner <joiner.onnx>")?,
            tokens: required(tokens, "--tokens <tokens.txt>")?,
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
