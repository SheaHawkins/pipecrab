//! [`LmStage`]: the generic adapter from any [`LanguageModel`] to a pipeline
//! [`Stage`], tracking the running [`Conversation`].

use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream::StreamExt;
use pipecrab_core::{
    DataFrame, Decision, Direction, Finality, Processor, Role, SystemFrame, Transcript,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{Conversation, GenParams, LanguageModel, LmError, Message, TokenOut};

/// Adapts any [`LanguageModel`] into a pipeline [`Stage`]: it accumulates the
/// running [`Conversation`], and on a **final user** [`Transcript`] it appends
/// the user's turn and generates a reply, streaming it downstream as agent
/// [`Transcript`]s — partials as the deltas arrive, then a final.
///
/// The system prompt is injected at construction and is the first message of the
/// conversation; every completed user utterance and generated reply is appended,
/// so successive generations see the whole history.
///
/// # State and the decide/perform split
///
/// The [`Conversation`] is the stage's state. Following the
/// [`Processor`]/[`Stage`] split, `decide_data` (sync, `&mut self`) appends the
/// user turn and emits a [`Generate`] effect; `perform` (`&self`) snapshots the
/// conversation, runs the generation, and — only *after* the final delta — locks
/// again to append the assistant turn. Because the conversation lives behind a
/// [`Mutex`] and is mutated only in synchronous critical sections (the
/// Mutex-after-await idiom), a barge-in that drops an in-flight `perform` leaves
/// no half-written turn.
///
/// # Barge-in
///
/// Each `.await` in `perform` — pulling the next delta and forwarding it — is a
/// point the run loop can drop `perform` at, so a barge-in
/// [`Interrupt`](SystemFrame::Interrupt) stops the reply within one delta. The
/// [`Interrupt`](SystemFrame::Interrupt) also reaches [`decide_system`], which
/// issues the [`cancel`](LanguageModel::cancel) control call so the engine's
/// worker stops decoding too.
///
/// # Known v1 limitation
///
/// An interrupted generation is **not recorded in the conversation at all**, even
/// though the user may have heard part of it: the assistant turn is appended only
/// after the final delta, so a dropped `perform` records nothing. Correct repair
/// needs a *played-up-to* marker from the audio sink (how much of the reply was
/// actually voiced before the barge-in) to record the truncated turn — that
/// marker does not exist yet, so it is out of scope here.
///
/// [`decide_data`]: Processor::decide_data
/// [`decide_system`]: Processor::decide_system
pub struct LmStage<M: LanguageModel> {
    model: M,
    params: GenParams,
    convo: Mutex<Conversation>,
}

impl<M: LanguageModel> LmStage<M> {
    /// Wrap `model` as a stage seeded with `system_prompt`, using default
    /// [`GenParams`].
    pub fn new(model: M, system_prompt: impl Into<std::sync::Arc<str>>) -> Self {
        Self::with_params(model, system_prompt, GenParams::default())
    }

    /// Wrap `model` as a stage seeded with `system_prompt` and explicit `params`.
    pub fn with_params(
        model: M,
        system_prompt: impl Into<std::sync::Arc<str>>,
        params: GenParams,
    ) -> Self {
        let convo = Conversation {
            messages: vec![Message::system(system_prompt)],
        };
        Self {
            model,
            params,
            convo: Mutex::new(convo),
        }
    }
}

/// Run a generation over the current conversation: [`LmStage`]'s
/// [`Processor::Effect`]. A unit — the conversation to generate from is read from
/// the stage's own state in `perform`, not carried in the effect.
pub struct Generate;

impl<M: LanguageModel> Processor for LmStage<M> {
    type Effect = Generate;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<Generate> {
        match frame {
            // The user's finished utterance: append it and generate a reply. The
            // text is Arc-backed, so this clone is a refcount bump.
            DataFrame::Transcript(Transcript {
                role: Role::User,
                finality: Finality::Final,
                text,
            }) => {
                self.convo
                    .lock()
                    .expect("LmStage conversation mutex poisoned")
                    .messages
                    .push(Message::user(text.clone()));
                Decision::drop().emit(Generate)
            }
            // An in-progress user transcript is not yet actionable in v1 — consume
            // it. Speculative prefill will hook here to warm the KV cache
            // from the partial before the final arrives.
            DataFrame::Transcript(Transcript {
                role: Role::User,
                finality: Finality::Partial { .. },
                ..
            }) => Decision::drop(),
            // Agent transcripts (our own output looping back), audio, custom
            // frames: not ours to consume.
            _ => Decision::forward(),
        }
    }

    fn decide_system(&mut self, _dir: Direction, frame: &SystemFrame) -> Decision<Generate> {
        // Barge-in: abort the engine's decode at once via the control call. The
        // run loop separately drops the in-flight `perform`; forwarding the
        // Interrupt lets downstream stages reset too.
        if matches!(frame, SystemFrame::Interrupt) {
            self.model.cancel();
        }
        Decision::forward()
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<M: LanguageModel> Stage for LmStage<M> {
    async fn perform(&self, _effect: Generate, out: &Outbound) -> Result<(), StageError> {
        // Snapshot the conversation under the lock, then release it before the
        // awaited generation — the guard must not cross an `.await`.
        let convo = {
            self.convo
                .lock()
                .expect("LmStage conversation mutex poisoned")
                .clone()
        };

        let mut stream = self.model.generate(&convo, &self.params).await?;
        let mut reply = String::new();
        // Each `.next().await` is a preemption point: a barge-in drops this future
        // here. Because the assistant turn is recorded only after the final delta
        // (below), a dropped generation leaves the conversation untouched.
        while let Some(item) = stream.next().await {
            let TokenOut { delta } = item?;
            reply.push_str(&delta);
            // LM output is append-only, so the whole accumulated reply is stable.
            let partial = Transcript::agent_partial(reply.clone());
            debug_assert!(
                matches!(partial.finality, Finality::Partial { stable } if stable == partial.text.len()),
                "agent partial must be append-only (stable == text.len())",
            );
            // Ignore the send error: it only happens once the sink has gone away
            // during shutdown, matching the runtime's own forward path.
            let _ = out.send_data(partial.into()).await;
        }
        let _ = out
            .send_data(Transcript::agent_final(reply.clone()).into())
            .await;

        // Record the assistant turn now — the Mutex-after-await idiom. This runs
        // synchronously after the final send's await resolves (no await between),
        // so a barge-in either drops us earlier — recording nothing — or lets the
        // whole turn commit; it never leaves a partially recorded message.
        self.convo
            .lock()
            .expect("LmStage conversation mutex poisoned")
            .messages
            .push(Message::assistant(reply));
        Ok(())
    }
}

impl From<LmError> for StageError {
    fn from(e: LmError) -> Self {
        // A failed generation is recoverable: skip this reply and keep the
        // pipeline alive. The run loop surfaces it as an Error frame upstream.
        StageError::new(e.to_string())
    }
}
