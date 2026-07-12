//! Adapts a [`StreamingTranscriber`] into a pipeline [`Stage`].

use async_trait::async_trait;
use pipecrab_core::{
    AudioChunk, AudioFormat, DataFrame, Decision, Direction, Processor, SystemFrame, Transcript,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{StreamingTranscriber, SttError, SttEvent};

/// Maps speech edges and audio to the [`StreamingTranscriber`] protocol.
///
/// `SpeechStarted` opens an utterance, each `Audio` chunk feeds it, and
/// `SpeechStopped` closes it.
/// The engine's [`SttEvent`]s are forwarded downstream as [`Transcript`]s.
///
/// # Format authority
///
/// The stage caches [`StreamingTranscriber::input_format`] and rejects a
/// mismatched chunk fatally.
///
/// # Protocol trust, not defense
///
/// The stage trusts the VAD edge-ordering contract. Protocol violations surface
/// as recoverable [`SttError::Engine`] errors.
///
/// # State and the decide/perform split
///
/// The stage is stateless. [`SystemFrame::Interrupt`] invokes
/// [`StreamingTranscriber::cancel`].
pub struct SttStage<S: StreamingTranscriber> {
    transcriber: S,
    /// The cached [`StreamingTranscriber::input_format`].
    expected: AudioFormat,
}

impl<S: StreamingTranscriber> SttStage<S> {
    /// Wrap `transcriber` as a stage, caching the format it declares.
    pub fn new(transcriber: S) -> Self {
        let expected = transcriber.input_format();
        Self {
            transcriber,
            expected,
        }
    }
}

/// One [`StreamingTranscriber`] protocol operation.
pub enum SttEffect {
    /// Open an utterance in the engine.
    Begin,
    /// Feed one window of audio to the open utterance and forward any events.
    Feed(AudioChunk),
    /// Close the utterance and drain remaining events.
    End,
    /// A chunk's format did not match the engine's; fail fatally.
    RejectFormat {
        /// The format of the rejected chunk.
        got: AudioFormat,
    },
}

impl<S: StreamingTranscriber> Processor for SttStage<S> {
    type Effect = SttEffect;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<SttEffect> {
        match frame {
            // Live speech: feed the chunk straight through. Unconditional — the
            // gate upstream guarantees only speech-time audio reaches us, so
            // there is no idle case to buffer for. The chunk is Arc-backed, so
            // this clone is a refcount bump.
            DataFrame::Audio(chunk) if chunk.format == self.expected => {
                Decision::drop().emit(SttEffect::Feed(chunk.clone()))
            }
            // Format-fatal admission: a mismatch is a wiring bug the engine can
            // never conform (it cannot detect rate from `&[f32]`). Cancel first
            // as hygiene — don't leave the worker mid-utterance — then reject.
            DataFrame::Audio(chunk) => {
                self.transcriber.cancel();
                Decision::drop().emit(SttEffect::RejectFormat { got: chunk.format })
            }
            // The gate opens the utterance: the onset audio is already bracketed
            // in behind this edge, so we can begin on the edge alone.
            DataFrame::SpeechStarted => Decision::forward().emit(SttEffect::Begin),
            // The gate closes the utterance: drain the final off the tail.
            DataFrame::SpeechStopped => Decision::forward().emit(SttEffect::End),
            // Transcripts, transport bytes, custom frames: not ours to touch.
            _ => Decision::forward(),
        }
    }

    fn decide_system(&mut self, _dir: Direction, frame: &SystemFrame) -> Decision<SttEffect> {
        match frame {
            SystemFrame::Interrupt => {
                // Control call (see the `Processor` control-call carve-out): flip
                // the engine's cancel flag right where the interrupt is decided,
                // so any in-flight utterance is abandoned promptly. Unconditional
                // because it is idempotent — a cancel while idle is a no-op.
                self.transcriber.cancel();
                Decision::forward()
            }
            // Start, Stop, Error, and any future frames: pass through untouched.
            _ => Decision::forward(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<S: StreamingTranscriber> Stage for SttStage<S> {
    async fn perform(&self, effect: SttEffect, out: &Outbound) -> Result<(), StageError> {
        match effect {
            SttEffect::Begin => {
                self.transcriber.begin_utterance().await?;
            }
            SttEffect::Feed(chunk) => {
                let events = self.transcriber.feed(&chunk.samples).await?;
                forward_events(events, out).await;
            }
            SttEffect::End => {
                let events = self.transcriber.end_utterance().await?;
                forward_events(events, out).await;
            }
            SttEffect::RejectFormat { got } => {
                return Err(StageError::fatal(format!(
                    "SttStage requires {} Hz/{} ch (declared by the engine); \
                     got {} Hz/{} ch — insert a resample stage upstream or \
                     reconfigure the source",
                    self.expected.sample_rate,
                    self.expected.channels,
                    got.sample_rate,
                    got.channels,
                )));
            }
        }
        Ok(())
    }
}

/// Forwards transcript events and ignores endpoint events.
async fn forward_events(events: Vec<SttEvent>, out: &Outbound) {
    for event in events {
        let transcript = match event {
            SttEvent::Partial { text, stable } => Transcript::user_partial(text, stable),
            SttEvent::Final(text) => Transcript::user_final(text),
            SttEvent::Endpoint => continue,
        };
        // Ignore the send error: it only fires once the sink is gone during
        // shutdown, matching the runtime's own forward path.
        let _ = out.send_data(transcript.into()).await;
    }
}

impl From<SttError> for StageError {
    fn from(e: SttError) -> Self {
        // A failed engine call is recoverable: skip it and keep the pipeline
        // alive. The run loop surfaces it as an Error frame upstream. Only the
        // format path (RejectFormat) is fatal.
        StageError::new(e.to_string())
    }
}
