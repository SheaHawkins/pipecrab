//! Adapts a [`LanguageModel`] into a pipeline [`Stage`].

use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream::StreamExt;
use pipecrab_core::{
    DataFrame, Decision, Direction, Finality, Processor, Role, SystemFrame, Transcript,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{Conversation, GenParams, LanguageModel, LmError, Message, TokenOut};

/// Converts final user [`Transcript`]s into streamed agent transcripts.
///
/// The stage retains the system prompt and completed turns in a [`Conversation`].
///
/// # State and the decide/perform split
///
/// [`Processor::decide_data`] appends a user turn. [`Stage::perform`] snapshots
/// the conversation and appends the assistant turn only after generation ends.
///
/// # Barge-in
///
/// [`SystemFrame::Interrupt`] drops the current effect and calls
/// [`LanguageModel::cancel`].
///
/// # Known v1 limitation
///
/// Interrupted generations are not added to the conversation.
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

/// Tells [`LmStage`] to generate from its current conversation.
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
