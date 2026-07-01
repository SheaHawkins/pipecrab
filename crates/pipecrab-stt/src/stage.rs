//! [`SttStage`]: the generic adapter from any [`Transcriber`] to a pipeline
//! [`Stage`].

use async_trait::async_trait;
use pipecrab_core::{AudioChunk, DataFrame, Decision, Processor};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{SttError, Transcriber};

/// Adapts any [`Transcriber`] into a pipeline [`Stage`]: on a
/// [`DataFrame::Audio`], it transcribes the chunk and emits a
/// [`DataFrame::Transcript`] in its place; every other frame passes through
/// untouched.
///
/// One utterance per audio frame for the spike — VAD segmentation (deciding
/// where an utterance begins and ends) is a separate stage upstream of this one.
///
/// Following the [`Processor`]/[`Stage`] split, [`decide_data`] only classifies
/// the frame and hands the chunk to `perform` as a [`Transcribe`] effect;
/// `perform` does the awaited inference. Because `perform` takes `&self` and the
/// transcriber owns *where* the work runs, a barge-in interrupt can drop an
/// in-flight transcription cleanly.
///
/// [`decide_data`]: Processor::decide_data
pub struct SttStage<T: Transcriber> {
    transcriber: T,
}

impl<T: Transcriber> SttStage<T> {
    /// Wrap `transcriber` as a stage.
    pub fn new(transcriber: T) -> Self {
        Self { transcriber }
    }
}

/// One audio chunk to transcribe: [`SttStage`]'s [`Processor::Effect`]. Emitted
/// by `decide_data`, interpreted by `perform`. Its inner chunk is private — only
/// the stage constructs one.
pub struct Transcribe(AudioChunk);

impl<T: Transcriber> Processor for SttStage<T> {
    type Effect = Transcribe;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<Transcribe> {
        match frame {
            // Consume the audio; its transcript replaces it downstream. The
            // chunk is Arc-backed, so this clone is a refcount bump.
            DataFrame::Audio(chunk) => Decision::drop().emit(Transcribe(chunk.clone())),
            // Transcripts, transport bytes, custom frames: not ours to touch.
            _ => Decision::forward(),
        }
    }
}

#[async_trait(?Send)]
impl<T: Transcriber> Stage for SttStage<T> {
    async fn perform(&self, Transcribe(chunk): Transcribe, out: &Outbound) -> Result<(), StageError> {
        let text = self.transcriber.transcribe(&chunk.samples, chunk.format).await?;
        // Ignore the send error: it only happens once the sink has gone away
        // during shutdown, matching the runtime's own forward path.
        let _ = out.send_data(DataFrame::Transcript(text.into())).await;
        Ok(())
    }
}

impl From<SttError> for StageError {
    fn from(e: SttError) -> Self {
        // A failed transcription is recoverable: skip this utterance and keep the
        // pipeline alive. The run loop surfaces it as an Error frame upstream.
        StageError::new(e.to_string())
    }
}
