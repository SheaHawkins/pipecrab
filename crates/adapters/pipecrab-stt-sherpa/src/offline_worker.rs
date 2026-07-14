use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};

use async_trait::async_trait;
use futures::channel::oneshot;
use pipecrab_core::AudioFormat;
use pipecrab_stt::{StreamingTranscriber, SttError, SttEvent};

use crate::config::{
    DEFAULT_MOONSHINE_CHUNK_DURATION, DEFAULT_MOONSHINE_CHUNK_OVERLAP, SAMPLE_RATE,
    duration_sample_count,
};
use crate::offline_backend::MoonshineV2Backend;
use crate::{MoonshineV2Config, OfflineBackend, SherpaSttBuildError};

const INPUT_FORMAT: AudioFormat = AudioFormat {
    sample_rate: SAMPLE_RATE,
    channels: 1,
};
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

#[derive(Clone, Copy)]
struct Chunking {
    samples: usize,
    overlap: usize,
}

impl Chunking {
    fn defaults() -> Result<Self, SherpaSttBuildError> {
        Ok(Self {
            samples: duration_sample_count("chunk_duration", DEFAULT_MOONSHINE_CHUNK_DURATION)?,
            overlap: duration_sample_count("chunk_overlap", DEFAULT_MOONSHINE_CHUNK_OVERLAP)?,
        })
    }
}

impl WorkerHandle {
    fn sender(&self) -> Result<&mpsc::Sender<Command>, SttError> {
        self.sender
            .as_ref()
            .ok_or_else(|| SttError::Engine("Sherpa offline STT worker is closed".into()))
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

/// A worker-backed Sherpa ONNX offline speech recognizer.
///
/// Audio chunks are accumulated on one dedicated actor thread. Ending an
/// utterance decodes one or more bounded, overlapping model windows and emits
/// exactly one merged final event; offline recognizers do not emit partial
/// hypotheses.
pub struct OfflineSherpaStt {
    generation: Arc<AtomicU64>,
    worker: WorkerHandle,
}

impl OfflineSherpaStt {
    /// Create a Moonshine v2 recognizer and actor thread.
    ///
    /// Model loading runs on the actor thread. This call waits for setup so a
    /// returned handle is ready to begin an utterance.
    pub fn new(config: MoonshineV2Config) -> Result<Self, SherpaSttBuildError> {
        config.validate()?;
        let (samples, overlap) = config.chunk_samples()?;
        Self::spawn(
            move || MoonshineV2Backend::create(config),
            Chunking { samples, overlap },
        )
    }

    /// Move a custom offline backend onto a new actor thread.
    pub fn with_backend(backend: impl OfflineBackend) -> Result<Self, SherpaSttBuildError> {
        Self::spawn(move || Ok(backend), Chunking::defaults()?)
    }

    fn spawn<B: OfflineBackend>(
        create: impl FnOnce() -> Result<B, SherpaSttBuildError> + Send + 'static,
        chunking: Chunking,
    ) -> Result<Self, SherpaSttBuildError> {
        let generation = Arc::new(AtomicU64::new(0));
        let worker = spawn_worker(generation.clone(), create, chunking)?;
        Ok(Self { generation, worker })
    }
}

#[async_trait]
impl StreamingTranscriber for OfflineSherpaStt {
    fn input_format(&self) -> AudioFormat {
        INPUT_FORMAT
    }

    async fn begin_utterance(&self) -> Result<(), SttError> {
        let generation = self.generation.load(Ordering::Acquire);
        let (reply, response) = oneshot::channel();
        self.worker
            .sender()?
            .send(Command::Begin { generation, reply })
            .map_err(|_| SttError::Engine("Sherpa offline STT worker stopped".into()))?;

        let (response_generation, result) = response.await.map_err(|_| {
            SttError::Engine("Sherpa offline STT worker dropped its response".into())
        })?;
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
            .map_err(|_| SttError::Engine("Sherpa offline STT worker stopped".into()))?;

        current_events_response(response, &self.generation).await
    }

    async fn end_utterance(&self) -> EventsResult {
        let generation = self.generation.load(Ordering::Acquire);
        let (reply, response) = oneshot::channel();
        self.worker
            .sender()?
            .send(Command::End { generation, reply })
            .map_err(|_| SttError::Engine("Sherpa offline STT worker stopped".into()))?;

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
        .map_err(|_| SttError::Engine("Sherpa offline STT worker dropped its response".into()))?;
    if response_generation != generation.load(Ordering::Acquire) {
        return Ok(Vec::new());
    }
    result
}

fn spawn_worker<B: OfflineBackend>(
    generation: Arc<AtomicU64>,
    create: impl FnOnce() -> Result<B, SherpaSttBuildError> + Send + 'static,
    chunking: Chunking,
) -> Result<WorkerHandle, SherpaSttBuildError> {
    let (sender, receiver) = mpsc::channel();
    let (setup_sender, setup_receiver) = mpsc::channel();
    let thread = thread::Builder::new()
        .name("pipecrab-stt-sherpa-offline".into())
        .spawn(move || match create() {
            Ok(recognizer) => {
                if setup_sender.send(Ok(())).is_ok() {
                    SttWorker::new(recognizer, &generation, chunking).run(receiver, &generation);
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

struct SttWorker<B: OfflineBackend> {
    recognizer: B,
    samples: Option<Vec<f32>>,
    generation: u64,
    chunking: Chunking,
}

impl<B: OfflineBackend> SttWorker<B> {
    fn new(recognizer: B, generation: &AtomicU64, chunking: Chunking) -> Self {
        Self {
            recognizer,
            samples: None,
            generation: generation.load(Ordering::Acquire),
            chunking,
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
        if self.samples.is_some() {
            return Err(SttError::Engine(
                "OfflineSherpaStt::begin_utterance called while an utterance is already active"
                    .into(),
            ));
        }
        self.samples = Some(Vec::new());
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
        let Some(buffer) = self.samples.as_mut() else {
            return Err(SttError::Engine(
                "OfflineSherpaStt::feed called without an active utterance".into(),
            ));
        };
        buffer.extend_from_slice(samples);
        Ok(Vec::new())
    }

    fn end(&mut self, command_generation: u64, generation: &AtomicU64) -> EventsResult {
        if !self.command_is_current(command_generation, generation) {
            return Ok(Vec::new());
        }
        let Some(samples) = self.samples.take() else {
            return Err(SttError::Engine(
                "OfflineSherpaStt::end_utterance called without an active utterance".into(),
            ));
        };

        let mut text = String::new();
        if samples.is_empty() {
            let next = self.recognizer.transcribe(&samples).unwrap_or_default();
            if !self.command_is_current(command_generation, generation) {
                return Ok(Vec::new());
            }
            merge_transcript(&mut text, &next);
        } else {
            let mut start = 0;
            loop {
                let end = samples.len().min(start + self.chunking.samples);
                let next = self
                    .recognizer
                    .transcribe(&samples[start..end])
                    .unwrap_or_default();
                if !self.command_is_current(command_generation, generation) {
                    return Ok(Vec::new());
                }
                merge_transcript(&mut text, &next);
                if end == samples.len() {
                    break;
                }
                start = end - self.chunking.overlap;
            }
        }
        Ok(vec![SttEvent::Final(text.into())])
    }

    fn command_is_current(&mut self, command_generation: u64, generation: &AtomicU64) -> bool {
        let current = generation.load(Ordering::Acquire);
        if current != self.generation {
            self.samples = None;
            self.generation = current;
        }
        command_generation == current
    }
}

fn merge_transcript(transcript: &mut String, next: &str) {
    let next = next.trim();
    if next.is_empty() {
        return;
    }
    if transcript.is_empty() {
        transcript.push_str(next);
        return;
    }
    if !transcript.chars().any(char::is_whitespace) && !next.chars().any(char::is_whitespace) {
        merge_unspaced_transcript(transcript, next);
        return;
    }

    let left_words: Vec<_> = transcript.split_whitespace().collect();
    let right_words: Vec<_> = next.split_whitespace().collect();
    let maximum = left_words.len().min(right_words.len());
    let overlap = (1..=maximum)
        .rev()
        .find(|&count| {
            left_words[left_words.len() - count..]
                .iter()
                .zip(&right_words[..count])
                .all(|(left, right)| normalized_tokens_match(left, right))
        })
        .unwrap_or(0);

    if overlap == 0 {
        transcript.push(' ');
        transcript.push_str(next);
        return;
    }

    let mut words: Vec<&str> = left_words[..left_words.len() - overlap].to_vec();
    for (left, right) in left_words[left_words.len() - overlap..]
        .iter()
        .zip(&right_words[..overlap])
    {
        let left_normalized = normalize_token(left);
        let right_normalized = normalize_token(right);
        words.push(
            if left_normalized.chars().count() < right_normalized.chars().count() {
                right
            } else {
                left
            },
        );
    }
    words.extend_from_slice(&right_words[overlap..]);
    *transcript = words.join(" ");
}

fn normalize_token(token: &str) -> String {
    token
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn normalized_tokens_match(left: &str, right: &str) -> bool {
    let left = normalize_token(left);
    let right = normalize_token(right);
    if left.is_empty() || right.is_empty() {
        return false;
    }
    left == right
        || (left.chars().count().min(right.chars().count()) >= 4
            && (left.starts_with(&right) || right.starts_with(&left)))
}

fn merge_unspaced_transcript(transcript: &mut String, next: &str) {
    let left: Vec<char> = transcript.chars().collect();
    let right: Vec<char> = next.chars().collect();
    let maximum = left.len().min(right.len());
    let overlap = (2..=maximum)
        .rev()
        .find(|&count| left[left.len() - count..] == right[..count])
        .unwrap_or(0);
    transcript.extend(right[overlap..].iter());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_repeated_window_words_without_losing_punctuation() {
        let mut transcript = "Hello, brave new world.".to_owned();

        merge_transcript(&mut transcript, "new world this is a test.");

        assert_eq!(transcript, "Hello, brave new world. this is a test.");
    }

    #[test]
    fn replaces_a_partial_boundary_word_with_the_complete_form() {
        let mut transcript = "the brothels after early night.".to_owned();

        merge_transcript(&mut transcript, "Early nightfall, the lamps lit.");

        assert_eq!(
            transcript,
            "the brothels after early nightfall, the lamps lit."
        );
    }

    #[test]
    fn merges_unspaced_transcripts_by_character_overlap() {
        let mut transcript = "这是一个语音测试".to_owned();

        merge_transcript(&mut transcript, "语音测试继续进行");

        assert_eq!(transcript, "这是一个语音测试继续进行");
    }

    #[test]
    fn appends_a_window_without_repeated_words() {
        let mut transcript = "first window".to_owned();

        merge_transcript(&mut transcript, "second window");

        assert_eq!(transcript, "first window second window");
    }
}
