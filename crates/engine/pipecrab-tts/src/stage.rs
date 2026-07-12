//! Adapts a [`Synthesizer`] into a pipeline [`Stage`].

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::StreamExt;
use pipecrab_core::{
    DataFrame, Decision, Direction, Finality, Processor, Role, SystemFrame, Transcript,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{Synthesizer, TtsError};

/// Converts final agent [`Transcript`]s into streamed [`Audio`](DataFrame::Audio).
///
/// Other frames pass through. A [`SentenceChunker`](crate::SentenceChunker)
/// upstream can turn a generation into sentence-sized final transcripts.
///
/// # Barge-in
///
/// [`Processor::decide_data`] emits a [`Speak`] effect without doing I/O.
/// [`SystemFrame::Interrupt`] drops that effect, calls [`Synthesizer::cancel`],
/// and flushes queued synthesized audio.
pub struct TtsStage<S: Synthesizer> {
    synth: S,
}

impl<S: Synthesizer> TtsStage<S> {
    /// Wrap `synth` as a stage.
    pub fn new(synth: S) -> Self {
        Self { synth }
    }
}

/// Text for [`TtsStage`] to synthesize.
pub struct Speak(Arc<str>);

impl<S: Synthesizer> Processor for TtsStage<S> {
    type Effect = Speak;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<Speak> {
        match frame {
            // The agent's finished speech: consume it, synthesize it in its
            // place. The text is Arc-backed, so this clone is a refcount bump.
            DataFrame::Transcript(Transcript {
                role: Role::Agent,
                finality: Finality::Final,
                text,
            }) => Decision::drop().emit(Speak(text.clone())),
            // User speech, agent partials, audio, custom frames: not ours to voice.
            _ => Decision::forward(),
        }
    }

    fn decide_system(&mut self, _dir: Direction, frame: &SystemFrame) -> Decision<Speak> {
        // Barge-in: stop the engine's worker at once via the control call. The
        // run loop separately drops the in-flight `perform`; forwarding the
        // Interrupt lets downstream stages reset too.
        if matches!(frame, SystemFrame::Interrupt) {
            self.synth.cancel();
        }
        Decision::forward()
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<S: Synthesizer> Stage for TtsStage<S> {
    async fn perform(&self, Speak(text): Speak, out: &Outbound) -> Result<(), StageError> {
        let mut s = self.synth.synthesize(&text).await?;
        // Each `.await` — pulling the next chunk and sending it — is where an
        // Interrupt can drop this future, stopping emission within one chunk.
        while let Some(chunk) = s.next().await {
            // Ignore the send error: it only happens once the sink has gone away
            // during shutdown, matching the runtime's own forward path.
            let _ = out.send_data(DataFrame::Audio(chunk?)).await;
        }
        Ok(())
    }
}

impl From<TtsError> for StageError {
    fn from(e: TtsError) -> Self {
        // A failed synthesis is recoverable: skip this utterance and keep the
        // pipeline alive. The run loop surfaces it as an Error frame upstream.
        StageError::new(e.to_string())
    }
}
