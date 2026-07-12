//! [`TtsStage`]: the generic adapter from any [`Synthesizer`] to a pipeline
//! [`Stage`].

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::StreamExt;
use pipecrab_core::{
    DataFrame, Decision, Direction, Finality, Processor, Role, SystemFrame, Transcript,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{Synthesizer, TtsError};

/// Adapts any [`Synthesizer`] into a pipeline [`Stage`]: on a **final agent**
/// [`Transcript`] it synthesizes the text and streams
/// [`Audio`](DataFrame::Audio) chunks in its place; every other frame passes
/// through untouched.
///
/// Only the agent's [`Final`](Finality::Final) speech is spoken — user speech
/// and in-progress agent partials are not the stage's to voice, so they forward.
/// A [`SentenceChunker`](crate::SentenceChunker) upstream turns a streaming
/// generation into per-sentence finals, so "final agent transcript" is often a
/// single sentence and playback starts without waiting for the whole reply.
///
/// # Barge-in
///
/// Following the [`Processor`]/[`Stage`] split, [`decide_data`] only classifies
/// the frame and hands the text to `perform` as a [`Speak`] effect; `perform`
/// pulls the synthesizer's stream and emits a chunk per item. Each `.await`
/// there is a point the run loop can drop `perform` at, so a barge-in
/// [`Interrupt`](SystemFrame::Interrupt) stops emission within one chunk. The
/// [`Interrupt`](SystemFrame::Interrupt) also reaches [`decide_system`], which
/// issues the [`cancel`](Synthesizer::cancel) control call so the engine's worker
/// stops producing too. Any [`Audio`](DataFrame::Audio) chunks already queued
/// downstream are discarded by the run loop's interrupt flush
/// ([`survives_flush`](DataFrame::survives_flush) is false for them).
///
/// [`decide_data`]: Processor::decide_data
/// [`decide_system`]: Processor::decide_system
pub struct TtsStage<S: Synthesizer> {
    synth: S,
}

impl<S: Synthesizer> TtsStage<S> {
    /// Wrap `synth` as a stage.
    pub fn new(synth: S) -> Self {
        Self { synth }
    }
}

/// One piece of text to synthesize: [`TtsStage`]'s [`Processor::Effect`].
/// Emitted by `decide_data`, interpreted by `perform`. Its inner text is private
/// — only the stage constructs one.
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
