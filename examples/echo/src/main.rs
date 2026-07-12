//! `echo` — capture your voice and play it straight back through a pipecrab
//! pipeline, so you can *hear* audio ride the pipeline end to end.
//!
//! ```text
//!   CpalSource ──▶ pump_in ──▶ [ EchoStage ] ──▶ pump_out ──▶ CpalSink
//!   (mic, RT)     (async)      (pipeline)         (async)      (speaker, RT)
//! ```
//!
//! The two pumps and the pipeline run as one future on a single thread
//! (`futures::executor::block_on(join!(..))`, tokio-free); cpal runs the device
//! callbacks on its own real-time OS threads, bridged to us by the backend's
//! lock-free ring buffers.
//!
//! # Running it
//!
//! ```console
//! $ cargo run -p echo                 # live monitor: hear yourself immediately
//! $ cargo run -p echo -- --delay-ms 400   # 400 ms delay: an audible echo
//! $ cargo run -p echo -- --seconds 5      # run for 5 s, then shut down cleanly
//! ```
//!
//! Use **headphones** — over speakers the mic re-captures the playback and you
//! get a feedback howl. On macOS the first run triggers a microphone-permission
//! prompt. Without `--seconds` it runs until you press Ctrl-C.

// The audio backend only exists on desktop targets; gate `main` to match.
#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
fn main() {
    if let Err(e) = desktop::run() {
        eprintln!("echo: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("echo requires a desktop OS (macOS, Windows, or Linux)");
    std::process::exit(1);
}

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
mod desktop {
    use std::collections::VecDeque;
    use std::error::Error;

    use async_trait::async_trait;
    use futures::executor::block_on;
    use pipecrab::{
        DataFrame, Decision, Direction, Outbound, PipelineBuilder, Processor, Received, Stage,
        StageError, SystemFrame,
    };
    use pipecrab_audio::{AudioChunk, AudioSink, AudioSource};
    use pipecrab_audio_cpal::{CpalConfig, CpalSink, CpalSource};

    /// Forwards `DataFrame::Audio` downstream, optionally delayed by a fixed
    /// backlog of chunks so playback is an audible echo instead of a live
    /// monitor.
    ///
    /// With `delay_chunks == 0` it is a pure pass-through (default
    /// [`Processor::decide_data`] behaviour). With a delay it holds each chunk
    /// in a queue and, once the queue is full, emits the oldest one — mutating
    /// the queue in `decide_data` and doing the send in `perform`, per the
    /// [`Processor`]/[`Stage`] split.
    struct EchoStage {
        delay_chunks: usize,
        backlog: VecDeque<AudioChunk>,
    }

    impl EchoStage {
        fn new(delay_chunks: usize) -> Self {
            Self {
                delay_chunks,
                backlog: VecDeque::new(),
            }
        }
    }

    /// One deferred audio chunk to send downstream.
    struct Play(AudioChunk);

    impl Processor for EchoStage {
        type Effect = Play;

        fn decide_data(&mut self, frame: &DataFrame) -> Decision<Play> {
            match frame {
                // No delay: forward the frame untouched (a live monitor).
                DataFrame::Audio(_) if self.delay_chunks == 0 => Decision::forward(),
                // Delayed: buffer this chunk; once the backlog is full, drop the
                // incoming frame and emit the oldest buffered chunk in its place.
                DataFrame::Audio(chunk) => {
                    self.backlog.push_back(chunk.clone());
                    match self.backlog.len() > self.delay_chunks {
                        true => Decision::drop().emit(Play(self.backlog.pop_front().unwrap())),
                        false => Decision::drop(), // still filling the delay buffer.
                    }
                }
                _ => Decision::forward(),
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl Stage for EchoStage {
        async fn perform(&self, Play(chunk): Play, out: &Outbound) -> Result<(), StageError> {
            // Ignore the send error: it only happens once the sink has gone away
            // during shutdown, matching the runtime's own forward path.
            let _ = out.send_data(DataFrame::Audio(chunk)).await;
            Ok(())
        }
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let Args {
            delay_ms,
            seconds: max_seconds,
        } = parse_args()?;

        // One config, shared by both ends; defaults to the system default
        // input/output devices with ~20 ms chunks.
        let config = CpalConfig::default();
        let source = CpalSource::new(&config)?;
        let sink = CpalSink::new(&config)?;

        // No resampling: capture and playback must agree on rate/channels. On a
        // shared-clock same-device setup they do; refuse the mismatch outright
        // rather than silently pitch-shifting the audio.
        if source.format() != sink.format() {
            return Err(format!(
                "device format mismatch: input is {} Hz/{} ch but output is {} Hz/{} ch; \
                 resampling is not supported — use matching devices",
                source.format().sample_rate,
                source.format().channels,
                sink.format().sample_rate,
                sink.format().channels,
            )
            .into());
        }

        let rate = source.format().sample_rate;
        let chunk_frames = source.chunk_frames();
        let chunk_ms = u64::from(config.chunk_ms);
        let delay_chunks = (delay_ms / chunk_ms) as usize;
        // ~chunk_ms per chunk, so this many chunks ≈ the requested run length.
        let max_chunks = max_seconds.map(|s| (s * 1000 / chunk_ms) as usize);

        println!("echo: in  = {} @ {} Hz mono", source.device_name(), rate);
        println!(
            "echo: out = {} @ {} Hz mono",
            sink.device_name(),
            sink.format().sample_rate
        );
        println!("echo: {chunk_frames} frames/chunk (~{chunk_ms} ms), delay {delay_ms} ms ({delay_chunks} chunks)");
        match max_chunks {
            Some(_) => println!("echo: running for {} s", max_seconds.unwrap()),
            None => println!("echo: running until Ctrl-C — use headphones!"),
        }

        let (ends, driver) = PipelineBuilder::new()
            .stage(EchoStage::new(delay_chunks))
            .build()
            .start();
        let input = ends.input;
        let output = ends.output;

        // In-pump: Start at boot, then each captured chunk as a typed Audio
        // frame. Returning (dropping `input`) closes the head and cascades
        // shutdown through the pipeline to `pump_out`.
        let pump_in = async move {
            let mut source = source;
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let mut sent = 0usize;
            loop {
                match source.next_chunk().await {
                    Ok(Some(chunk)) => {
                        if input.send_data(DataFrame::Audio(chunk)).await.is_err() {
                            break; // downstream gone.
                        }
                        sent += 1;
                        if max_chunks.is_some_and(|max| sent >= max) {
                            break; // bounded run elapsed; drop `input` to shut down.
                        }
                    }
                    Ok(None) => break, // source exhausted (a live mic never does).
                    Err(e) => {
                        eprintln!("echo: capture stopped: {e}");
                        break;
                    }
                }
            }
        };

        // Out-pump: play Audio frames to the speaker; log-and-ignore the Start.
        // Exhaustive match, no downcast.
        let pump_out = async move {
            let mut sink = sink;
            let mut output = output;
            while let Some(received) = output.recv().await {
                match received {
                    Received::Data(DataFrame::Audio(chunk)) => {
                        if let Err(e) = sink.play(chunk).await {
                            eprintln!("echo: playback stopped: {e}");
                            break;
                        }
                    }
                    Received::Data(_) => {} // no other media in this pipeline.
                    Received::Sys(_, _) => {} // lifecycle frames: nothing to do here.
                }
            }
        };

        // One thread drives the pipeline and both pumps; cpal's device threads
        // run alongside. Tokio-free, consistent with the runtime.
        block_on(async { futures::join!(driver, pump_in, pump_out) });
        println!("echo: stopped.");
        Ok(())
    }

    /// Parsed command-line options.
    struct Args {
        delay_ms: u64,
        seconds: Option<u64>,
    }

    /// Parse the process arguments, erroring on an unknown flag or a
    /// missing/invalid value rather than silently ignoring it.
    fn parse_args() -> Result<Args, String> {
        let mut delay_ms = 0;
        let mut seconds = None;
        let mut args = std::env::args().skip(1); // skip argv[0].
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--delay-ms" => delay_ms = parse_value(&mut args, "--delay-ms")?,
                "--seconds" => seconds = Some(parse_value(&mut args, "--seconds")?),
                other => {
                    return Err(format!(
                        "unknown argument {other:?} (expected --delay-ms or --seconds)"
                    ))
                }
            }
        }
        Ok(Args { delay_ms, seconds })
    }

    /// Take the next argument as a `u64`, erroring if it is missing or not a
    /// non-negative integer.
    fn parse_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<u64, String> {
        let raw = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        raw.parse()
            .map_err(|_| format!("{flag} expects a non-negative integer, got {raw:?}"))
    }
}
