use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};

use async_trait::async_trait;
use futures::StreamExt;
use futures::channel::mpsc as futures_mpsc;
use pipecrab_core::{AudioChunk, AudioFormat};
use pipecrab_tts::{Synthesizer, TtsAudioStream, TtsError};

use crate::backend::KokoroBackend;
use crate::{Backend, KokoroConfig, SherpaTtsBuildError};

type ChunkSender = futures_mpsc::UnboundedSender<Result<AudioChunk, TtsError>>;

enum Command {
    Synthesize {
        epoch: u64,
        text: String,
        output: ChunkSender,
    },
}

struct WorkerHandle {
    sender: Option<mpsc::Sender<Command>>,
    thread: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    fn sender(&self) -> Result<&mpsc::Sender<Command>, TtsError> {
        self.sender
            .as_ref()
            .ok_or_else(|| TtsError::Engine("Sherpa TTS worker is closed".into()))
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A worker-backed Sherpa ONNX offline text-to-speech engine.
///
/// Synthesis runs on one dedicated actor thread. Sherpa generates sentence by
/// sentence, and each sentence's samples are forwarded to the returned
/// [`TtsAudioStream`] as soon as the engine produces them, so playback of a
/// long reply starts after its first sentence.
///
/// [`cancel`](Synthesizer::cancel) advances an epoch the in-flight generation
/// checks between sentences: the engine stops producing within one sentence,
/// and stale output never reaches a later stream. Dropping the stream stops
/// the engine the same way.
pub struct SherpaTts {
    epoch: Arc<AtomicU64>,
    format: AudioFormat,
    worker: WorkerHandle,
}

impl SherpaTts {
    /// Create a Kokoro engine and its actor thread.
    ///
    /// Model loading runs on the actor thread. This call waits for setup so a
    /// returned handle is ready to synthesize and already knows the model's
    /// output format.
    pub fn new(config: KokoroConfig) -> Result<Self, SherpaTtsBuildError> {
        config.validate()?;
        Self::spawn(move || KokoroBackend::create(config))
    }

    /// Move a custom backend onto a new actor thread.
    pub fn with_backend(backend: impl Backend) -> Result<Self, SherpaTtsBuildError> {
        Self::spawn(move || Ok(backend))
    }

    fn spawn<B: Backend>(
        create: impl FnOnce() -> Result<B, SherpaTtsBuildError> + Send + 'static,
    ) -> Result<Self, SherpaTtsBuildError> {
        let epoch = Arc::new(AtomicU64::new(0));
        let worker_epoch = Arc::clone(&epoch);
        let (sender, receiver) = mpsc::channel();
        let (setup_sender, setup_receiver) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("pipecrab-tts-sherpa".into())
            .spawn(move || match create() {
                Ok(mut backend) => {
                    let sample_rate = backend.sample_rate();
                    if sample_rate == 0 {
                        let _ = setup_sender.send(Err(SherpaTtsBuildError::CreateEngine(
                            "engine reported a zero output sample rate".into(),
                        )));
                        return;
                    }
                    let format = AudioFormat::new(sample_rate, 1);
                    if setup_sender.send(Ok(format)).is_ok() {
                        run(backend, receiver, &worker_epoch, format);
                    }
                }
                Err(error) => {
                    let _ = setup_sender.send(Err(error));
                }
            })
            .map_err(|error| SherpaTtsBuildError::Worker(format!("spawn thread: {error}")))?;

        match setup_receiver.recv() {
            Ok(Ok(format)) => Ok(Self {
                epoch,
                format,
                worker: WorkerHandle {
                    sender: Some(sender),
                    thread: Some(thread),
                },
            }),
            Ok(Err(error)) => {
                let _ = thread.join();
                Err(error)
            }
            Err(_) => {
                let _ = thread.join();
                Err(SherpaTtsBuildError::Worker(
                    "thread exited before reporting setup".into(),
                ))
            }
        }
    }
}

impl Drop for SherpaTts {
    fn drop(&mut self) {
        // Stop any in-flight generation before WorkerHandle joins the thread,
        // so shutdown is bounded by one sentence rather than a whole utterance.
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }
}

#[async_trait]
impl Synthesizer for SherpaTts {
    fn output_format(&self) -> AudioFormat {
        self.format
    }

    async fn synthesize(&self, text: &str) -> Result<TtsAudioStream, TtsError> {
        // Taking a fresh epoch stops any stale in-flight generation, so this
        // synthesis starts clean even if a prior stream was abandoned.
        let epoch = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        let (output, receiver) = futures_mpsc::unbounded();
        self.worker
            .sender()?
            .send(Command::Synthesize {
                epoch,
                text: text.into(),
                output,
            })
            .map_err(|_| TtsError::Engine("Sherpa TTS worker stopped".into()))?;
        Ok(receiver.boxed())
    }

    fn cancel(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }
}

fn run<B: Backend>(
    mut backend: B,
    receiver: mpsc::Receiver<Command>,
    epoch: &Arc<AtomicU64>,
    format: AudioFormat,
) {
    while let Ok(Command::Synthesize {
        epoch: command_epoch,
        text,
        output,
    }) = receiver.recv()
    {
        if epoch.load(Ordering::Acquire) != command_epoch {
            continue; // canceled before it started; the stream just ends.
        }
        let emit_epoch = Arc::clone(epoch);
        let emit_output = output.clone();
        let emit = Box::new(move |samples: &[f32]| {
            if emit_epoch.load(Ordering::Acquire) != command_epoch {
                return false; // barge-in: tell the engine to stop producing.
            }
            if samples.is_empty() {
                return true;
            }
            let chunk = AudioChunk::new(Arc::from(samples), format);
            // A dropped stream is the other stop signal: the stage stopped
            // pulling, so stop the engine too.
            emit_output.unbounded_send(Ok(chunk)).is_ok()
        });
        if let Err(error) = backend.generate(&text, emit) {
            if epoch.load(Ordering::Acquire) == command_epoch {
                let _ = output.unbounded_send(Err(TtsError::Engine(error)));
            }
        }
        // `output` and the callback's clone drop here, closing the stream.
    }
}
