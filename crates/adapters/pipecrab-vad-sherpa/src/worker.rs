use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};

use async_trait::async_trait;
use futures::channel::oneshot;
use pipecrab_core::AudioFormat;
use pipecrab_vad::{VadError, VadEvent, VoiceActivityDetector};

use crate::backend::SherpaBackend;
use crate::config::{SAMPLE_RATE, WINDOW_SAMPLES};
use crate::{Backend, SherpaVadBuildError, SherpaVadConfig};

const INPUT_FORMAT: AudioFormat = AudioFormat {
    sample_rate: SAMPLE_RATE,
    channels: 1,
};

type ProcessResult = Result<Vec<VadEvent>, VadError>;

enum Command {
    Process {
        samples: Arc<[f32]>,
        generation: u64,
        reply: oneshot::Sender<(u64, ProcessResult)>,
    },
}

struct WorkerHandle {
    sender: Option<mpsc::Sender<Command>>,
    thread: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    fn sender(&self) -> Result<&mpsc::Sender<Command>, VadError> {
        self.sender
            .as_ref()
            .ok_or_else(|| VadError::Engine("Sherpa VAD worker is closed".into()))
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

/// A worker-backed Sherpa ONNX Silero voice-activity detector.
///
/// The handle is `Send + Sync`; the native detector is not shared. One actor
/// thread owns it and serializes all access. Input may have any length and is
/// accumulated into exact 512-sample windows at 16 kHz mono.
pub struct SherpaVad {
    generation: Arc<AtomicU64>,
    worker: WorkerHandle,
}

impl SherpaVad {
    /// Create a detector and its actor thread from a Silero model configuration.
    ///
    /// Model loading runs on the actor thread. This call waits for setup so a
    /// returned handle is ready to process audio.
    pub fn new(config: SherpaVadConfig) -> Result<Self, SherpaVadBuildError> {
        Self::spawn(move || {
            SherpaBackend::create(config).map(|backend| Box::new(backend) as Box<dyn Backend>)
        })
    }

    /// Move a custom backend onto a new actor thread.
    ///
    /// This is useful for deterministic engines and tests. The worker takes
    /// exclusive ownership before the constructor returns.
    pub fn with_backend(backend: impl Backend) -> Result<Self, SherpaVadBuildError> {
        Self::spawn(move || Ok(Box::new(backend)))
    }

    fn spawn(
        create: impl FnOnce() -> Result<Box<dyn Backend>, SherpaVadBuildError> + Send + 'static,
    ) -> Result<Self, SherpaVadBuildError> {
        let generation = Arc::new(AtomicU64::new(0));
        let worker = spawn_worker(generation.clone(), create)?;
        Ok(Self { generation, worker })
    }
}

#[async_trait]
impl VoiceActivityDetector for SherpaVad {
    fn input_format(&self) -> AudioFormat {
        INPUT_FORMAT
    }

    async fn process(&self, samples: Arc<[f32]>) -> ProcessResult {
        let generation = self.generation.load(Ordering::Acquire);
        let (reply, response) = oneshot::channel();
        self.worker
            .sender()?
            .send(Command::Process {
                samples,
                generation,
                reply,
            })
            .map_err(|_| VadError::Engine("Sherpa VAD worker stopped".into()))?;

        let (response_generation, result) = response
            .await
            .map_err(|_| VadError::Engine("Sherpa VAD worker dropped its response".into()))?;
        if response_generation != self.generation.load(Ordering::Acquire) {
            return Ok(Vec::new());
        }
        result
    }

    fn reset(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }
}

fn spawn_worker(
    generation: Arc<AtomicU64>,
    create: impl FnOnce() -> Result<Box<dyn Backend>, SherpaVadBuildError> + Send + 'static,
) -> Result<WorkerHandle, SherpaVadBuildError> {
    let (sender, receiver) = mpsc::channel();
    let (setup_sender, setup_receiver) = mpsc::channel();
    let thread = thread::Builder::new()
        .name("pipecrab-vad-sherpa".into())
        .spawn(move || match create() {
            Ok(backend) => {
                if setup_sender.send(Ok(())).is_ok() {
                    run_worker(backend, receiver, generation);
                }
            }
            Err(error) => {
                let _ = setup_sender.send(Err(error));
            }
        })
        .map_err(|error| SherpaVadBuildError::Worker(format!("spawn thread: {error}")))?;

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
            Err(SherpaVadBuildError::Worker(
                "thread exited before reporting setup".into(),
            ))
        }
    }
}

fn run_worker(
    mut backend: Box<dyn Backend>,
    receiver: mpsc::Receiver<Command>,
    reset_generation: Arc<AtomicU64>,
) {
    let mut remainder = Vec::with_capacity(WINDOW_SAMPLES);
    let mut observed_generation = reset_generation.load(Ordering::Acquire);

    while let Ok(command) = receiver.recv() {
        match command {
            Command::Process {
                samples,
                generation,
                reply,
            } => {
                let events = process_samples(
                    backend.as_mut(),
                    &mut remainder,
                    &samples,
                    generation,
                    &reset_generation,
                    &mut observed_generation,
                );
                let result = Ok(events.unwrap_or_default());
                let _ = reply.send((generation, result));
            }
        }
    }
}

fn process_samples(
    backend: &mut dyn Backend,
    remainder: &mut Vec<f32>,
    samples: &[f32],
    command_generation: u64,
    reset_generation: &AtomicU64,
    observed_generation: &mut u64,
) -> Option<Vec<VadEvent>> {
    if !generation_is_current(
        backend,
        remainder,
        command_generation,
        reset_generation,
        observed_generation,
    ) {
        return None;
    }

    let mut events = Vec::new();
    let mut offset = 0;
    if !remainder.is_empty() {
        let take = (WINDOW_SAMPLES - remainder.len()).min(samples.len());
        remainder.extend_from_slice(&samples[..take]);
        offset = take;
        if remainder.len() == WINDOW_SAMPLES {
            if !generation_is_current(
                backend,
                remainder,
                command_generation,
                reset_generation,
                observed_generation,
            ) {
                return None;
            }
            process_window(backend, remainder, &mut events);
            remainder.clear();
        }
    }

    let mut windows = samples[offset..].chunks_exact(WINDOW_SAMPLES);
    for window in &mut windows {
        if !generation_is_current(
            backend,
            remainder,
            command_generation,
            reset_generation,
            observed_generation,
        ) {
            return None;
        }
        process_window(backend, window, &mut events);
    }

    if !generation_is_current(
        backend,
        remainder,
        command_generation,
        reset_generation,
        observed_generation,
    ) {
        return None;
    }
    remainder.extend_from_slice(windows.remainder());

    generation_is_current(
        backend,
        remainder,
        command_generation,
        reset_generation,
        observed_generation,
    )
    .then_some(events)
}

fn generation_is_current(
    backend: &mut dyn Backend,
    remainder: &mut Vec<f32>,
    command_generation: u64,
    reset_generation: &AtomicU64,
    observed_generation: &mut u64,
) -> bool {
    let current = reset_generation.load(Ordering::Acquire);
    if current != *observed_generation {
        backend.reset();
        remainder.clear();
        *observed_generation = current;
    }
    current == command_generation
}

fn process_window(backend: &mut dyn Backend, window: &[f32], events: &mut Vec<VadEvent>) {
    debug_assert_eq!(window.len(), WINDOW_SAMPLES);
    let before = backend.detected();
    backend.accept_waveform(window);
    let after = backend.detected();
    match (before, after) {
        (false, true) => events.push(VadEvent::SpeechStarted),
        (true, false) => events.push(VadEvent::SpeechStopped),
        _ => {}
    }
    while !backend.is_empty() {
        backend.pop();
    }
}
