//! Turn policy for pipecrab: [`BargeInStage`] turns user speech onset into a
//! downstream [`Interrupt`](SystemFrame::Interrupt).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use async_trait::async_trait;
use pipecrab_core::{DataFrame, Decision, Direction, Processor, SystemFrame};
use pipecrab_runtime::{Outbound, Stage, StageError};

/// Emits a downstream [`Interrupt`](SystemFrame::Interrupt) on every
/// [`SpeechStarted`](DataFrame::SpeechStarted), then re-emits the edge behind
/// it.
///
/// Place it immediately below the STT stage. The data lane is downstream-only
/// and nothing routes up, so the stage cannot know whether the agent is
/// speaking — it interrupts unconditionally. That is harmless while the
/// pipeline is idle: every stage's interrupt handling is an idempotent control
/// call, and flushing an empty queue is a no-op.
///
/// The Interrupt is sent *before* the edge is re-emitted. The interrupt flush
/// is causal — frames queued at or after the Interrupt survive it — so
/// consuming the edge and re-emitting it behind the Interrupt is what keeps
/// the barge-in utterance from being destroyed by its own interrupt.
pub struct BargeInStage;

impl BargeInStage {
    /// A stage with no state.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BargeInStage {
    fn default() -> Self {
        Self::new()
    }
}

/// Interrupt the agent, then re-open the user's turn: [`BargeInStage`]'s
/// [`Processor::Effect`].
pub struct BargeIn;

impl Processor for BargeInStage {
    type Effect = BargeIn;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<BargeIn> {
        match frame {
            // Consume the edge; perform re-emits it behind the Interrupt.
            DataFrame::SpeechStarted => Decision::drop().emit(BargeIn),
            _ => Decision::forward(),
        }
    }
    // decide_system: default forward — an Interrupt from further upstream
    // passes through untouched.
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Stage for BargeInStage {
    async fn perform(&self, _effect: BargeIn, out: &Outbound) -> Result<(), StageError> {
        let _ = out.send_system(Direction::Down, SystemFrame::Interrupt).await;
        let _ = out.send_data(DataFrame::SpeechStarted).await;
        Ok(())
    }
}
