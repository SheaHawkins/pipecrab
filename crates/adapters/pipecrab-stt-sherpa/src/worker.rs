use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};

use async_trait::async_trait;
use futures::channel::oneshot;
use pipecrab_core::AudioFormat;
use pipecrab_stt::{StreamingTranscriber, SttError, SttEvent};

use crate::backend::SherpaBackend;
use crate::config::SAMPLE_RATE;
use crate::{Backend, SherpaSttBuildError, SherpaSttConfig};

const INPUT_FORMAT: AudioFormat = AudioFormat {
    sample_rate: SAMPLE_RATE,
    channels: 1,
};
const FINAL_PADDING_SAMPLES: usize = SAMPLE_RATE as usize * 3 / 10;
static FINAL_PADDING: [f32; FINAL_PADDING_SAMPLES] = [0.0; FINAL_PADDING_SAMPLES];

type EventsResult = Result<Vec<SttEvent>, SttError>;
type BeginReply = oneshot::Sender<(u64, Result<(), SttError>)>;
type EventsReply = oneshot::Sender<(u64, EventsResult)>;

enum Command {
    Begin {
        generation: u64,
        reply: BeginReply,
    },
    Feed {
        samples: Arc<[f32]>,
        generation: u64,
        reply: EventsReply,
    },
    End {
        generation: u64,
        reply: EventsReply,
    },
}

struct WorkerHandle {
    sender: Option<mpsc::Sender<Command>>,
    thread: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    fn sender(&self) -> Result<&mpsc::Sender<Command>, SttError> {
        self.sender
            .as_ref()
            .ok_or_else(|| SttError::Engine("Sherpa STT worker is closed".into()))
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

/// A worker-backed Sherpa ONNX streaming speech recognizer.
///
/// The handle is `Send + Sync`; the native recognizer and active stream stay on
/// one dedicated actor thread. PipeCrab VAD edges open and close each stream,
/// and changed Sherpa hypotheses are emitted with no stable prefix.
pub struct SherpaStt {
    generation: Arc<AtomicU64>,
    worker: WorkerHandle,
}

impl SherpaStt {
    /// Create a recognizer and actor thread from a streaming transducer
    /// configuration.
    ///
    /// Model loading runs on the actor thread. This call waits for setup so a
    /// returned handle is ready to begin an utterance.
    pub fn new(config: SherpaSttConfig) -> Result<Self, SherpaSttBuildError> {
        config.validate()?;
        Self::spawn(move || SherpaBackend::create(config))
    }

    /// Move a custom backend onto a new actor thread.
    ///
    /// This supports deterministic recognizers and tests without changing the
    /// worker ownership or utterance protocol.
    pub fn with_backend(backend: impl Backend) -> Result<Self, SherpaSttBuildError> {
        Self::spawn(move || Ok(backend))
    }

    fn spawn<B: Backend>(
        create: impl FnOnce() -> Result<B, SherpaSttBuildError> + Send + 'static,
    ) -> Result<Self, SherpaSttBuildError> {
        let generation = Arc::new(AtomicU64::new(0));
        let worker = spawn_worker(generation.clone(), create)?;
        Ok(Self { generation, worker })
    }
}

#[async_trait]
impl StreamingTranscriber for SherpaStt {
    fn input_format(&self) -> AudioFormat {
        INPUT_FORMAT
    }

    async fn begin_utterance(&self) -> Result<(), SttError> {
        let generation = self.generation.load(Ordering::Acquire);
        let (reply, response) = oneshot::channel();
        self.worker
            .sender()?
            .send(Command::Begin { generation, reply })
            .map_err(|_| SttError::Engine("Sherpa STT worker stopped".into()))?;

        let (response_generation, result) = response
            .await
            .map_err(|_| SttError::Engine("Sherpa STT worker dropped its response".into()))?;
        if response_generation != self.generation.load(Ordering::Acquire) {
            return Ok(());
        }
        result
    }

    async fn feed(&self, samples: Arc<[f32]>) -> EventsResult {
        let generation = self.generation.load(Ordering::Acquire);
        let (reply, response) = oneshot::channel();
        self.worker
            .sender()?
            .send(Command::Feed {
                samples,
                generation,
                reply,
            })
            .map_err(|_| SttError::Engine("Sherpa STT worker stopped".into()))?;

        current_events_response(response, &self.generation).await
    }

    async fn end_utterance(&self) -> EventsResult {
        let generation = self.generation.load(Ordering::Acquire);
        let (reply, response) = oneshot::channel();
        self.worker
            .sender()?
            .send(Command::End { generation, reply })
            .map_err(|_| SttError::Engine("Sherpa STT worker stopped".into()))?;

        current_events_response(response, &self.generation).await
    }

    fn cancel(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }
}

async fn current_events_response(
    response: oneshot::Receiver<(u64, EventsResult)>,
    generation: &AtomicU64,
) -> EventsResult {
    let (response_generation, result) = response
        .await
        .map_err(|_| SttError::Engine("Sherpa STT worker dropped its response".into()))?;
    if response_generation != generation.load(Ordering::Acquire) {
        return Ok(Vec::new());
    }
    result
}

fn spawn_worker<B: Backend>(
    generation: Arc<AtomicU64>,
    create: impl FnOnce() -> Result<B, SherpaSttBuildError> + Send + 'static,
) -> Result<WorkerHandle, SherpaSttBuildError> {
    let (sender, receiver) = mpsc::channel();
    let (setup_sender, setup_receiver) = mpsc::channel();
    let thread = thread::Builder::new()
        .name("pipecrab-stt-sherpa".into())
        .spawn(move || match create() {
            Ok(recognizer) => {
                if setup_sender.send(Ok(())).is_ok() {
                    SttWorker::new(recognizer, &generation).run(receiver, &generation);
                }
            }
            Err(error) => {
                let _ = setup_sender.send(Err(error));
            }
        })
        .map_err(|error| SherpaSttBuildError::Worker(format!("spawn thread: {error}")))?;

    match setup_receiver.recv() {
        Ok(Ok(())) => Ok(WorkerHandle {
            sender: Some(sender),
            thread: Some(thread),
        }),
        Ok(Err(error)) => {
            let _ = thread.join();
            Err(error)
        }
        Err(_) => {
            let _ = thread.join();
            Err(SherpaSttBuildError::Worker(
                "thread exited before reporting setup".into(),
            ))
        }
    }
}

struct SttWorker<B: Backend> {
    recognizer: B,
    stream: Option<B::Stream>,
    generation: u64,
    last_partial: String,
}

impl<B: Backend> Drop for SttWorker<B> {
    fn drop(&mut self) {
        self.stream = None;
    }
}

impl<B: Backend> SttWorker<B> {
    fn new(recognizer: B, generation: &AtomicU64) -> Self {
        Self {
            recognizer,
            stream: None,
            generation: generation.load(Ordering::Acquire),
            last_partial: String::new(),
        }
    }

    fn run(mut self, receiver: mpsc::Receiver<Command>, generation: &AtomicU64) {
        while let Ok(command) = receiver.recv() {
            match command {
                Command::Begin {
                    generation: command_generation,
                    reply,
                } => {
                    let result = self.begin(command_generation, generation);
                    let _ = reply.send((command_generation, result));
                }
                Command::Feed {
                    samples,
                    generation: command_generation,
                    reply,
                } => {
                    let result = self.feed(&samples, command_generation, generation);
                    let _ = reply.send((command_generation, result));
                }
                Command::End {
                    generation: command_generation,
                    reply,
                } => {
                    let result = self.end(command_generation, generation);
                    let _ = reply.send((command_generation, result));
                }
            }
        }
    }

    fn begin(&mut self, command_generation: u64, generation: &AtomicU64) -> Result<(), SttError> {
        if !self.command_is_current(command_generation, generation) {
            return Ok(());
        }
        if self.stream.is_some() {
            return Err(SttError::Engine(
                "SherpaStt::begin_utterance called while an utterance is already active".into(),
            ));
        }

        self.last_partial.clear();
        self.stream = Some(self.recognizer.create_stream());
        Ok(())
    }

    fn feed(
        &mut self,
        samples: &[f32],
        command_generation: u64,
        generation: &AtomicU64,
    ) -> EventsResult {
        if !self.command_is_current(command_generation, generation) {
            return Ok(Vec::new());
        }
        let Some(mut stream) = self.stream.take() else {
            return Err(SttError::Engine(
                "SherpaStt::feed called without an active utterance".into(),
            ));
        };

        self.recognizer.accept_waveform(&mut stream, samples);
        if !self.local_stream_is_current(command_generation, generation) {
            return Ok(Vec::new());
        }

        while self.recognizer.is_ready(&mut stream) {
            if !self.local_stream_is_current(command_generation, generation) {
                return Ok(Vec::new());
            }
            self.recognizer.decode(&mut stream);
            if !self.local_stream_is_current(command_generation, generation) {
                return Ok(Vec::new());
            }
        }

        let text = self.recognizer.get_result(&mut stream).unwrap_or_default();
        if !self.local_stream_is_current(command_generation, generation) {
            return Ok(Vec::new());
        }

        let events = if text == self.last_partial {
            Vec::new()
        } else {
            self.last_partial.clone_from(&text);
            vec![SttEvent::Partial {
                text: text.into(),
                stable: 0,
            }]
        };
        self.stream = Some(stream);
        Ok(events)
    }

    fn end(&mut self, command_generation: u64, generation: &AtomicU64) -> EventsResult {
        if !self.command_is_current(command_generation, generation) {
            return Ok(Vec::new());
        }
        let Some(mut stream) = self.stream.take() else {
            return Err(SttError::Engine(
                "SherpaStt::end_utterance called without an active utterance".into(),
            ));
        };

        self.recognizer.accept_waveform(&mut stream, &FINAL_PADDING);
        if !self.local_stream_is_current(command_generation, generation) {
            return Ok(Vec::new());
        }

        self.recognizer.input_finished(&mut stream);
        if !self.local_stream_is_current(command_generation, generation) {
            return Ok(Vec::new());
        }

        while self.recognizer.is_ready(&mut stream) {
            if !self.local_stream_is_current(command_generation, generation) {
                return Ok(Vec::new());
            }
            self.recognizer.decode(&mut stream);
            if !self.local_stream_is_current(command_generation, generation) {
                return Ok(Vec::new());
            }
        }

        let final_text = self.recognizer.get_result(&mut stream).unwrap_or_default();
        if !self.local_stream_is_current(command_generation, generation) {
            return Ok(Vec::new());
        }

        self.last_partial.clear();
        drop(stream);
        Ok(vec![SttEvent::Final(final_text.into())])
    }

    fn command_is_current(&mut self, command_generation: u64, generation: &AtomicU64) -> bool {
        let current = generation.load(Ordering::Acquire);
        if current != self.generation {
            self.drop_active_stream();
            self.generation = current;
        }
        command_generation == current
    }

    fn local_stream_is_current(&mut self, command_generation: u64, generation: &AtomicU64) -> bool {
        let current = generation.load(Ordering::Acquire);
        if current != self.generation {
            self.last_partial.clear();
            self.generation = current;
        }
        command_generation == current
    }

    fn drop_active_stream(&mut self) {
        self.stream = None;
        self.last_partial.clear();
    }
}
