use std::sync::{Arc, Mutex, mpsc};
use std::thread::ThreadId;

use futures::StreamExt;
use futures::executor::block_on;
use pipecrab_core::{AudioChunk, AudioFormat};
use pipecrab_tts::{Synthesizer, TtsError};
use pipecrab_tts_sherpa::{Backend, Emit, KokoroConfig, SherpaTts};

#[derive(Default)]
struct Probe {
    texts: Vec<String>,
    threads: Vec<ThreadId>,
}

struct ScriptedBackend {
    chunks: Vec<Vec<f32>>,
    probe: Arc<Mutex<Probe>>,
}

impl Backend for ScriptedBackend {
    fn sample_rate(&mut self) -> u32 {
        22_050
    }

    fn generate(&mut self, text: &str, mut emit: Emit) -> Result<(), String> {
        let mut probe = self.probe.lock().unwrap();
        probe.texts.push(text.to_owned());
        probe.threads.push(std::thread::current().id());
        drop(probe);
        for chunk in &self.chunks {
            if !emit(chunk) {
                break;
            }
        }
        Ok(())
    }
}

fn collect(synth: &SherpaTts, text: &str) -> Vec<Result<AudioChunk, TtsError>> {
    block_on(async {
        let stream = synth.synthesize(text).await.unwrap();
        stream.collect().await
    })
}

#[test]
fn streams_each_generated_segment_with_the_engine_format() {
    let probe = Arc::new(Mutex::new(Probe::default()));
    let synth = SherpaTts::with_backend(ScriptedBackend {
        // The empty segment mirrors an engine flushing nothing; it must be
        // skipped rather than surfaced as a zero-length chunk.
        chunks: vec![vec![0.1, 0.2], vec![], vec![0.3]],
        probe: probe.clone(),
    })
    .unwrap();

    assert_eq!(synth.output_format(), AudioFormat::new(22_050, 1));
    let chunks: Vec<AudioChunk> = collect(&synth, "hello kokoro")
        .into_iter()
        .map(|chunk| chunk.unwrap())
        .collect();

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].samples.as_ref(), [0.1, 0.2]);
    assert_eq!(chunks[1].samples.as_ref(), [0.3]);
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk.format == AudioFormat::new(22_050, 1))
    );

    let probe = probe.lock().unwrap();
    assert_eq!(probe.texts, ["hello kokoro"]);
    assert_ne!(probe.threads[0], std::thread::current().id());
}

#[test]
fn synthesizes_consecutive_requests() {
    let probe = Arc::new(Mutex::new(Probe::default()));
    let synth = SherpaTts::with_backend(ScriptedBackend {
        chunks: vec![vec![0.5]],
        probe: probe.clone(),
    })
    .unwrap();

    assert_eq!(collect(&synth, "one").len(), 1);
    assert_eq!(collect(&synth, "two").len(), 1);
    assert_eq!(probe.lock().unwrap().texts, ["one", "two"]);
}

struct BlockingBackend {
    reached: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
    done: mpsc::Sender<Vec<bool>>,
}

impl Backend for BlockingBackend {
    fn sample_rate(&mut self) -> u32 {
        24_000
    }

    fn generate(&mut self, _text: &str, mut emit: Emit) -> Result<(), String> {
        let first = emit(&[0.1, 0.2]);
        self.reached.send(()).unwrap();
        self.release.recv().unwrap();
        let second = emit(&[0.3]);
        self.done.send(vec![first, second]).unwrap();
        Ok(())
    }
}

fn blocking_synth() -> (
    SherpaTts,
    mpsc::Receiver<()>,
    mpsc::Sender<()>,
    mpsc::Receiver<Vec<bool>>,
) {
    let (reached_tx, reached_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let synth = SherpaTts::with_backend(BlockingBackend {
        reached: reached_tx,
        release: release_rx,
        done: done_tx,
    })
    .unwrap();
    (synth, reached_rx, release_tx, done_rx)
}

#[test]
fn cancellation_stops_an_in_flight_generation() {
    let (synth, reached, release, done) = blocking_synth();

    let stream = block_on(synth.synthesize("hello")).unwrap();
    reached.recv().unwrap(); // the engine has produced its first segment.
    synth.cancel();
    release.send(()).unwrap();

    // The engine was told to stop at the first post-cancel emission.
    assert_eq!(done.recv().unwrap(), [true, false]);
    // The pre-cancel segment was already queued; nothing follows it.
    let chunks = block_on(stream.collect::<Vec<_>>());
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].as_ref().unwrap().samples.as_ref(), [0.1, 0.2]);
}

#[test]
fn dropping_the_stream_stops_the_engine() {
    let (synth, reached, release, done) = blocking_synth();

    let stream = block_on(synth.synthesize("hello")).unwrap();
    reached.recv().unwrap();
    drop(stream); // the stage stopped pulling — e.g. its perform was dropped.
    release.send(()).unwrap();

    assert_eq!(done.recv().unwrap(), [true, false]);
}

struct FailingBackend;

impl Backend for FailingBackend {
    fn sample_rate(&mut self) -> u32 {
        24_000
    }

    fn generate(&mut self, _text: &str, _emit: Emit) -> Result<(), String> {
        Err("model exploded".into())
    }
}

#[test]
fn engine_errors_surface_on_the_stream() {
    let synth = SherpaTts::with_backend(FailingBackend).unwrap();

    let results = collect(&synth, "hello");

    assert_eq!(results, [Err(TtsError::Engine("model exploded".into()))]);
}

struct ZeroRateBackend;

impl Backend for ZeroRateBackend {
    fn sample_rate(&mut self) -> u32 {
        0
    }

    fn generate(&mut self, _text: &str, _emit: Emit) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn a_zero_sample_rate_engine_fails_setup() {
    let error = match SherpaTts::with_backend(ZeroRateBackend) {
        Ok(_) => panic!("a zero sample rate must fail setup"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("zero output sample rate"));
}

#[test]
#[ignore = "requires SHERPA_KOKORO_MODEL, SHERPA_KOKORO_VOICES, SHERPA_KOKORO_TOKENS, and SHERPA_KOKORO_DATA_DIR"]
fn kokoro_synthesizes_speech() {
    let config = KokoroConfig::new(
        std::env::var("SHERPA_KOKORO_MODEL").expect("set SHERPA_KOKORO_MODEL"),
        std::env::var("SHERPA_KOKORO_VOICES").expect("set SHERPA_KOKORO_VOICES"),
        std::env::var("SHERPA_KOKORO_TOKENS").expect("set SHERPA_KOKORO_TOKENS"),
        std::env::var("SHERPA_KOKORO_DATA_DIR").expect("set SHERPA_KOKORO_DATA_DIR"),
    );
    let synth = SherpaTts::new(config).unwrap();

    let format = synth.output_format();
    assert_eq!(format.channels, 1);
    assert!(format.sample_rate > 0);

    let chunks: Vec<AudioChunk> = collect(&synth, "Hello from pipecrab. This is Kokoro speaking.")
        .into_iter()
        .map(|chunk| chunk.expect("synthesis must not error"))
        .collect();

    assert!(!chunks.is_empty(), "known text produced no audio");
    let samples: usize = chunks.iter().map(|chunk| chunk.samples.len()).sum();
    // Two spoken sentences are comfortably longer than half a second.
    assert!(
        samples > format.sample_rate as usize / 2,
        "synthesis produced only {samples} samples"
    );
    assert!(chunks.iter().all(|chunk| chunk.format == format));
    println!(
        "kokoro: {} chunks, {:.2} s @ {} Hz",
        chunks.len(),
        samples as f32 / format.sample_rate as f32,
        format.sample_rate
    );
}
