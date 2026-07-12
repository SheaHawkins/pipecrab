//! [`SttStage`]: the stateless protocol adapter from any [`StreamingTranscriber`]
//! to a pipeline [`Stage`], driven by the VAD's speech edges.

use async_trait::async_trait;
use pipecrab_core::{
    AudioChunk, AudioFormat, DataFrame, Decision, Direction, Processor, SystemFrame, Transcript,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{SttError, SttEvent, StreamingTranscriber};

/// Adapts any [`StreamingTranscriber`] into a pipeline [`Stage`] as a **stateless
/// protocol adapter**.
///
/// So the stage is a pure translation: `SpeechStarted` → open the utterance,
/// each `Audio` chunk → feed it, `SpeechStopped` → close it and drain the final.
/// The engine's [`SttEvent`]s are forwarded downstream as [`Transcript`]s.
///
/// # Format authority
///
/// The engine *declares* the one format it accepts via
/// [`input_format`](StreamingTranscriber::input_format); the stage caches it in
/// [`new`](Self::new) and enforces it. A nonconforming chunk is a wiring bug,
/// and the stage rejects it **fatally** (a resample stage upstream is the fix)
/// rather than running deaf. See the crate-level [format authority](crate) note.
///
/// # Protocol trust, not defense
///
/// Edge alternation and edges-bracket-audio are the gate's *documented
/// invariants*. The stage trusts them: it does not track whether an utterance is
/// open, so a `Feed` before a `Begin` (or a double `Begin`) is not silently
/// absorbed — the engine's own protocol errors ([`Buffered`](crate::Buffered)
/// produces them; native adapters will too) surface as recoverable
/// [`Engine`](SttError::Engine) stage errors, loud and attributable.
///
/// # State and the decide/perform split
///
/// Following the [`Processor`]/[`Stage`] split, `decide_*` is synchronous and
/// `perform` drives the awaited engine calls. Here `decide_data` touches **no**
/// mutable state — the `&mut self` goes unused. The engine itself is a 
//  worker-handle (see [`StreamingTranscriber`]), so a barge-in that
/// drops an in-flight [`feed`](StreamingTranscriber::feed) leaves no torn state,
/// and the [`Interrupt`](SystemFrame::Interrupt) cancel is a *control call*
/// invoked right where the interrupt is decided.
pub struct SttStage<S: StreamingTranscriber> {
    transcriber: S,
    /// The one format the engine accepts, cached from
    /// [`input_format`](StreamingTranscriber::input_format) in [`new`](Self::new).
    expected: AudioFormat,
}

impl<S: StreamingTranscriber> SttStage<S> {
    /// Wrap `transcriber` as a stage, caching the format it declares.
    pub fn new(transcriber: S) -> Self {
        let expected = transcriber.input_format();
        Self { transcriber, expected }
    }
}

/// One step of the utterance protocol: [`SttStage`]'s [`Processor::Effect`].
/// Emitted by `decide_*`, interpreted by `perform`.
pub enum SttEffect {
    /// Open an utterance in the engine.
    Begin,
    /// Feed one window of audio to the open utterance and forward any events.
    Feed(AudioChunk),
    /// Close the utterance, draining its remaining events (including the final).
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

/// Forward each engine event downstream as a [`Transcript`]. `Endpoint` is a v1
/// no-op — the engine's own end-of-utterance signal has no frame to map to yet
/// (a future `TurnEnded` is out of scope), so it is ignored. TODO: turns
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
