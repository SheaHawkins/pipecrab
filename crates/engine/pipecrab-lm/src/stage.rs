//! [`LmStage`]: the generic adapter from any [`LanguageModel`] to a pipeline
//! [`Stage`], tracking the running [`Conversation`] and turning a model's
//! [`ModelDelta`] stream into native [`ModelFrame`]s and agent transcripts.

use std::collections::HashSet;
use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream::StreamExt;
use pipecrab_core::{
    DataFrame, Decision, Direction, Finality, ModelFrame, ModelInput, Processor, Role, SystemFrame,
    ToolCall, Transcript,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

use crate::{
    Conversation, GenParams, LanguageModel, LmConfigError, LmError, Message, ModelDelta,
    ToolDefinition,
};

/// Adapts any [`LanguageModel`] into a pipeline [`Stage`]: it accumulates the
/// running [`Conversation`] and, when a turn triggers generation, converts the
/// model's [`ModelDelta`] stream into native frames — visible text as agent
/// [`Transcript`]s (partials as the deltas arrive, then a final), tool calls as
/// [`ModelFrame::ToolCall`], each generation bracketed by
/// [`GenerationStarted`](ModelFrame::GenerationStarted) /
/// [`GenerationFinished`](ModelFrame::GenerationFinished).
///
/// # Inputs
///
/// Three inputs drive a turn:
///
/// * A **final user** [`Transcript`] appends a user message and generates.
/// * [`ModelInput::Context`](pipecrab_core::ModelInput::Context) appends a
///   non-user message *without* generating (background context for a later turn).
/// * [`ModelInput::Respond`](pipecrab_core::ModelInput::Respond) appends a
///   non-user message *and* generates.
///
/// The system prompt is injected at construction as the first message.
///
/// # Tools
///
/// Tools configured via [`with_tools`](Self::with_tools) /
/// [`add_tools`](Self::add_tools) are validated once (duplicate names rejected)
/// and passed to every generation. An adapter that wraps a higher-level agent
/// (e.g. Rig) keeps its own registered tools internal; the stage neither reads nor
/// copies them.
///
/// # State and the decide/perform split
///
/// The [`Conversation`] is the stage's state. `decide_data` (sync, `&mut self`)
/// appends the incoming turn and emits a [`Generate`] effect where a generation
/// is wanted; `perform` (`&self`) snapshots the conversation, runs the
/// generation, and — only *after* the stream completes — locks again to append
/// the structured assistant turn (visible text plus every tool call). Because the
/// conversation is mutated only in synchronous critical sections (the
/// Mutex-after-await idiom), a barge-in that drops an in-flight `perform` leaves
/// no half-written turn.
///
/// # Barge-in
///
/// Each `.await` in `perform` is a point the run loop can drop `perform` at, so a
/// barge-in [`Interrupt`](SystemFrame::Interrupt) stops the reply within one
/// delta. The interrupt also reaches [`decide_system`](Processor::decide_system),
/// which issues the [`cancel`](LanguageModel::cancel) control call so the engine's
/// worker stops decoding too. An interrupted generation commits no assistant turn
/// and emits no [`GenerationFinished`](ModelFrame::GenerationFinished); the
/// [`ToolCall`](ModelFrame::ToolCall) frames already emitted are ordinary
/// surviving pipeline data.
pub struct LmStage<M: LanguageModel> {
    model: M,
    params: GenParams,
    tools: Vec<ToolDefinition>,
    convo: Mutex<Conversation>,
}

impl<M: LanguageModel> LmStage<M> {
    /// Wrap `model` as a stage seeded with `system_prompt`, using default
    /// [`GenParams`] and no stage tools.
    pub fn new(model: M, system_prompt: impl Into<std::sync::Arc<str>>) -> Self {
        Self::build(model, system_prompt, GenParams::default())
    }

    /// Wrap `model` as a stage seeded with `system_prompt` and explicit `params`.
    pub fn with_params(
        model: M,
        system_prompt: impl Into<std::sync::Arc<str>>,
        params: GenParams,
    ) -> Self {
        Self::build(model, system_prompt, params)
    }

    /// Wrap `model` as a stage seeded with `system_prompt` and stage `tools`.
    ///
    /// Fails if the set has an invalid or duplicate tool.
    pub fn with_tools(
        model: M,
        system_prompt: impl Into<std::sync::Arc<str>>,
        tools: impl IntoIterator<Item = ToolDefinition>,
    ) -> Result<Self, LmConfigError> {
        Self::build(model, system_prompt, GenParams::default()).add_tools(tools)
    }

    /// Add stage tools to the effective set, revalidating it.
    ///
    /// Fails if any resulting tool is invalid or two share a name.
    pub fn add_tools(
        mut self,
        tools: impl IntoIterator<Item = ToolDefinition>,
    ) -> Result<Self, LmConfigError> {
        self.tools.extend(tools);
        validate_tool_set(&self.tools)?;
        Ok(self)
    }

    /// The stage tools passed to every generation.
    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    /// Seed a stage with the system prompt and no tools.
    fn build(model: M, system_prompt: impl Into<std::sync::Arc<str>>, params: GenParams) -> Self {
        Self {
            convo: Mutex::new(Conversation {
                messages: vec![Message::system(system_prompt)],
            }),
            tools: Vec::new(),
            params,
            model,
        }
    }

    /// Append a message to conversation history under the lock.
    fn append(&self, message: Message) {
        self.convo
            .lock()
            .expect("LmStage conversation mutex poisoned")
            .messages
            .push(message);
    }
}

/// Reject an empty name, a non-object schema, or a duplicate name anywhere in the
/// effective tool set. The tool fields are public, so the set is revalidated
/// rather than trusting each [`ToolDefinition`] was built through
/// [`ToolDefinition::new`].
fn validate_tool_set(tools: &[ToolDefinition]) -> Result<(), LmConfigError> {
    let mut seen: HashSet<&str> = HashSet::with_capacity(tools.len());
    for tool in tools {
        if tool.name.is_empty() {
            return Err(LmConfigError::EmptyToolName);
        }
        if !tool.parameters.is_object() {
            return Err(LmConfigError::ToolParametersNotObject {
                name: tool.name.clone(),
            });
        }
        if !seen.insert(tool.name.as_ref()) {
            return Err(LmConfigError::DuplicateToolName {
                name: tool.name.clone(),
            });
        }
    }
    Ok(())
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
                self.append(Message::user(text.clone()));
                Decision::drop().emit(Generate)
            }
            // An in-progress user transcript is not yet actionable in v1 — consume
            // it. Speculative prefill will hook here to warm the KV cache from the
            // partial before the final arrives.
            DataFrame::Transcript(Transcript {
                role: Role::User,
                finality: Finality::Partial { .. },
                ..
            }) => Decision::drop(),
            // Non-user context: append it, but do not generate.
            DataFrame::Model(ModelFrame::Input(ModelInput::Context(message))) => {
                self.append(Message::from(message.clone()));
                Decision::drop()
            }
            // Non-user input that warrants a reply: append it and generate.
            DataFrame::Model(ModelFrame::Input(ModelInput::Respond(message))) => {
                self.append(Message::from(message.clone()));
                Decision::drop().emit(Generate)
            }
            // Agent transcripts (our own output looping back), our own generation
            // frames, audio, custom frames: not ours to consume.
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

        // Only start the generation boundary once the stream is in hand: a
        // failed start emits nothing.
        let mut stream = self
            .model
            .generate(&convo, &self.params, &self.tools)
            .await?;
        let _ = out.send_data(ModelFrame::GenerationStarted.into()).await;

        let mut reply = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        // Each `.next().await` and each send is a preemption point: a barge-in
        // drops this future here. Because the assistant turn is committed only
        // after the stream completes (below), a dropped generation leaves the
        // conversation untouched — though any tool call already emitted survives.
        while let Some(item) = stream.next().await {
            match item? {
                ModelDelta::Text(delta) => {
                    reply.push_str(&delta);
                    // LM output is append-only, so the whole accumulated reply is
                    // stable.
                    let partial = Transcript::agent_partial(reply.clone());
                    debug_assert!(
                        matches!(partial.finality, Finality::Partial { stable } if stable == partial.text.len()),
                        "agent partial must be append-only (stable == text.len())",
                    );
                    // Ignore the send error: it only happens once the sink has gone
                    // away during shutdown, matching the runtime's own forward path.
                    let _ = out.send_data(partial.into()).await;
                }
                ModelDelta::ToolCall(call) => {
                    tool_calls.push(call.clone());
                    let _ = out.send_data(ModelFrame::ToolCall(call).into()).await;
                }
            }
        }

        // No empty final transcript — a tool-call-only turn emits none.
        if !reply.is_empty() {
            let _ = out
                .send_data(Transcript::agent_final(reply.clone()).into())
                .await;
        }
        let _ = out.send_data(ModelFrame::GenerationFinished.into()).await;

        // Commit the structured assistant turn now — the Mutex-after-await idiom.
        // This runs synchronously after the final send's await resolves (no await
        // between), so a barge-in either drops us earlier — committing nothing — or
        // lets the whole turn commit; it never leaves a partially recorded message.
        self.append(Message::assistant_with_tool_calls(reply, tool_calls));
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
