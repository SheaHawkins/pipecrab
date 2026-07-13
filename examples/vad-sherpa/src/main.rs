//! Print Sherpa VAD speech edges from the default microphone.
//!
//! ```console
//! cargo run -p vad-sherpa -- --model ./silero_vad.onnx
//! cargo run -p vad-sherpa -- --model ./silero_vad.onnx --seconds 10
//! ```

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
fn main() {
    if let Err(error) = desktop::run() {
        eprintln!("vad-sherpa: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("vad-sherpa requires a desktop OS (macOS, Windows, or Linux)");
    std::process::exit(1);
}

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
mod desktop {
    use std::error::Error;
    use std::path::PathBuf;

    use futures::executor::block_on;
    use pipecrab::{DataFrame, Direction, PipelineBuilder, Received, SystemFrame};
    use pipecrab_audio::{AudioFormat, AudioSource, ResamplerStage};
    use pipecrab_audio_cpal::{CpalConfig, CpalSource};
    use pipecrab_vad::VadStage;
    use pipecrab_vad_sherpa::{SherpaVad, SherpaVadConfig};

    const SHERPA_FORMAT: AudioFormat = AudioFormat {
        sample_rate: 16_000,
        channels: 1,
    };

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let audio_config = CpalConfig::default();
        let source = CpalSource::new(&audio_config)?;
        let detector = SherpaVad::new(SherpaVadConfig::new(args.model))?;
        let resampler = ResamplerStage::new(SHERPA_FORMAT)?;

        println!(
            "vad-sherpa: input = {} @ {} Hz mono",
            source.device_name(),
            source.format().sample_rate
        );
        println!("vad-sherpa: processing @ 16000 Hz mono");
        match args.seconds {
            Some(seconds) => println!("vad-sherpa: listening for {seconds} seconds"),
            None => println!("vad-sherpa: listening until Ctrl-C"),
        }

        let max_chunks = args
            .seconds
            .map(|seconds| (seconds * 1000 / u64::from(audio_config.chunk_ms)) as usize);
        let (ends, driver) = PipelineBuilder::new()
            .stage(resampler)
            .stage(VadStage::new(detector))
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
                        eprintln!("vad-sherpa: capture stopped: {error}");
                        break;
                    }
                }
            }
        };

        let drain = async move {
            while let Some(received) = output.recv().await {
                match received {
                    Received::Data(DataFrame::SpeechStarted) => println!("SpeechStarted"),
                    Received::Data(DataFrame::SpeechStopped) => println!("SpeechStopped"),
                    Received::Data(_) => {}
                    Received::Sys(_, _) => {}
                }
            }
        };

        block_on(async { futures::join!(driver, pump_in, drain) });
        println!("vad-sherpa: stopped");
        Ok(())
    }

    struct Args {
        model: PathBuf,
        seconds: Option<u64>,
    }

    fn parse_args() -> Result<Args, String> {
        let mut model = None;
        let mut seconds = None;
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--model" => model = Some(PathBuf::from(value(&mut args, "--model")?)),
                "--seconds" => {
                    let raw = value(&mut args, "--seconds")?;
                    seconds = Some(raw.parse().map_err(|_| {
                        format!("--seconds expects a non-negative integer, got {raw:?}")
                    })?);
                }
                other => {
                    return Err(format!(
                        "unknown argument {other:?} (expected --model or --seconds)"
                    ));
                }
            }
        }
        let model = model.ok_or_else(|| "--model <silero_vad.onnx> is required".to_string())?;
        Ok(Args { model, seconds })
    }

    fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
        args.next().ok_or_else(|| format!("{flag} needs a value"))
    }
}
