//! `LmStage` adapts a `LanguageModel` into a stage: it appends the triggering
//! turn to the running conversation and translates the model's `ModelDelta`
//! stream into native frames — cumulative agent transcript partials then a final
//! for text, `ModelFrame::ToolCall` for calls, bracketed by
//! `GenerationStarted`/`GenerationFinished` — committing one structured assistant
//! turn only once the stream completes.
//!
//! Frame ordering and barge-in go through the real pipeline; the conversation
//! invariants are pinned at the `decide`/`perform` level, where the (private)
//! conversation is observed indirectly via the scripted mock's record of what
//! each later generation was asked to generate from.
//!
//! Deterministic and tokio-free (`block_on`), so it rides the default
//! `cargo test --workspace` path.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::executor::block_on;
use futures::future::FutureExt;
use futures::sink::SinkExt;
use futures::stream::StreamExt;
use pipecrab_core::{
    DataFrame, Decision, Direction, Disposition, Finality, ModelFrame, ModelInput, ModelMessage,
    Processor, Role, SystemFrame, Transcript,
};
use pipecrab_lm::{
    Conversation, GenParams, Generate, LanguageModel, LmConfigError, LmError, LmStage, Message,
    ModelDelta, ModelStream, ToolDefinition,
};
use pipecrab_runtime::{PipelineBuilder, Received, Stage, link};
use serde_json::json;

// --- A scripted LanguageModel mock. ------------------------------------------

/// What a [`ScriptedLm`] observed, shared with the test via an `Arc`.
#[derive(Default)]
struct LmProbe {
    cancels: AtomicUsize,
    seen: Mutex<Vec<Conversation>>,
    tools_seen: Mutex<Vec<Vec<ToolDefinition>>>,
}

impl LmProbe {
    /// How many times [`LanguageModel::cancel`] has been called.
    fn cancels(&self) -> usize {
        self.cancels.load(Ordering::SeqCst)
    }

    /// A snapshot of the conversations passed to `generate`, in call order.
    /// Inspecting a later call's conversation is how a test sees what earlier
    /// turns were recorded.
    fn seen(&self) -> Vec<Conversation> {
        self.seen.lock().unwrap().clone()
    }

    /// The effective tool sets passed to `generate`, in call order.
    fn tools_seen(&self) -> Vec<Vec<ToolDefinition>> {
        self.tools_seen.lock().unwrap().clone()
    }
}

/// One `unfold` step of a parked generation stream.
enum Step {
    /// Emit this delta, then advance to [`Step::Park`].
    Emit(ModelDelta, mpsc::Sender<()>, oneshot::Receiver<()>),
    /// Signal the test that the stream has parked, then wait to be released — a
    /// barge-in drops this future (cancelling `block`) before it resolves.
    Park(mpsc::Sender<()>, oneshot::Receiver<()>),
}

/// A hardware-free [`LanguageModel`]: yields scripted deltas, records every call,
/// exposes intrinsic tools, and — when built with [`parking`](ScriptedLm::parking)
/// — parks its first generation after one delta so a barge-in can drop it.
struct ScriptedLm {
    deltas: Vec<ModelDelta>,
    intrinsic: Vec<ToolDefinition>,
    /// `Some` until the first (parking) generation consumes it; later generations
    /// finish normally.
    park: Mutex<Option<(mpsc::Sender<()>, oneshot::Receiver<()>)>>,
    probe: Arc<LmProbe>,
}

impl ScriptedLm {
    /// A model that yields `deltas` and completes on every generation.
    fn finishing<I>(deltas: I) -> (Self, Arc<LmProbe>)
    where
        I: IntoIterator<Item = ModelDelta>,
    {
        Self::with_intrinsic(deltas, Vec::new())
    }

    /// A finishing model that also advertises `intrinsic` tool definitions.
    fn with_intrinsic<I>(deltas: I, intrinsic: Vec<ToolDefinition>) -> (Self, Arc<LmProbe>)
    where
        I: IntoIterator<Item = ModelDelta>,
    {
        let probe = Arc::new(LmProbe::default());
        let me = Self {
            deltas: deltas.into_iter().collect(),
            intrinsic,
            park: Mutex::new(None),
            probe: probe.clone(),
        };
        (me, probe)
    }

    /// A model whose **first** generation yields one delta, signals on `reached`,
    /// then parks on `block` (a barge-in drops the future here); every later
    /// generation finishes normally.
    fn parking<I>(
        deltas: I,
        reached: mpsc::Sender<()>,
        block: oneshot::Receiver<()>,
    ) -> (Self, Arc<LmProbe>)
    where
        I: IntoIterator<Item = ModelDelta>,
    {
        let probe = Arc::new(LmProbe::default());
        let me = Self {
            deltas: deltas.into_iter().collect(),
            intrinsic: Vec::new(),
            park: Mutex::new(Some((reached, block))),
            probe: probe.clone(),
        };
        (me, probe)
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl LanguageModel for ScriptedLm {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.intrinsic.clone()
    }

    async fn generate(
        &self,
        convo: &Conversation,
        _params: &GenParams,
        tools: &[ToolDefinition],
    ) -> Result<ModelStream, LmError> {
        self.probe.seen.lock().unwrap().push(convo.clone());
        self.probe.tools_seen.lock().unwrap().push(tools.to_vec());
        let deltas = self.deltas.clone();
        // The first generation parks (if this mock was built to); later ones run
        // to completion.
        if let Some((reached, block)) = self.park.lock().unwrap().take() {
            let init = match deltas.into_iter().next() {
                Some(first) => Step::Emit(first, reached, block),
                None => Step::Park(reached, block),
            };
            let stream = futures::stream::unfold(init, |step| async move {
                match step {
                    Step::Emit(delta, reached, block) => {
                        Some((Ok(delta), Step::Park(reached, block)))
                    }
                    Step::Park(mut reached, block) => {
                        let _ = reached.send(()).await;
                        let _ = block.await;
                        None
                    }
                }
            });
            Ok(stream.boxed())
        } else {
            let items: Vec<Result<ModelDelta, LmError>> = deltas.into_iter().map(Ok).collect();
            Ok(futures::stream::iter(items).boxed())
        }
    }

    fn cancel(&self) {
        self.probe.cancels.fetch_add(1, Ordering::SeqCst);
    }

    async fn save_state(&self) -> Result<Vec<u8>, LmError> {
        Ok(Vec::new())
    }

    async fn load_state(&self, _blob: &[u8]) -> Result<(), LmError> {
        Ok(())
    }
}

// --- Helpers. ----------------------------------------------------------------

/// A text delta.
fn text(s: &str) -> ModelDelta {
    ModelDelta::Text(Arc::from(s))
}

/// A tool-call delta with empty arguments.
fn call(id: &str, name: &str) -> ModelDelta {
    ModelDelta::tool_call(id, name, json!({})).expect("valid tool call")
}

/// A tool definition with a trivial object schema.
fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(name, "desc", json!({ "type": "object" })).expect("valid tool")
}

/// A final user transcript as a data frame — the input that drives a generation.
fn user_final(s: &str) -> DataFrame {
    Transcript::user_final(s).into()
}

/// Extract the single `Generate` a triggering turn must emit, asserting the input
/// is consumed rather than forwarded.
fn take_generate(d: Decision<Generate>) -> Generate {
    assert_eq!(d.disposition, Disposition::Drop, "a triggering turn is consumed");
    d.effects
        .into_iter()
        .next()
        .expect("a triggering turn emits one Generate")
}

/// Drive one generation over a fresh channel and return the frames it emitted, in
/// order.
async fn run_generation(stage: &LmStage<ScriptedLm>, effect: Generate) -> Vec<DataFrame> {
    let (out, mut inbound) = link(64);
    stage.perform(effect, &out).await.unwrap();
    drop(out); // close the channel so the drain terminates
    let mut frames = Vec::new();
    while let Some(received) = inbound.recv().await {
        if let Received::Data(frame) = received {
            frames.push(frame);
        }
    }
    frames
}

/// Classify a frame into a compact tag for order assertions.
#[derive(Debug, PartialEq, Eq)]
enum Tag {
    Started,
    Finished,
    Call(String),
    Partial(String),
    Final(String),
}

fn tag(frame: &DataFrame) -> Tag {
    match frame {
        DataFrame::Model(ModelFrame::GenerationStarted) => Tag::Started,
        DataFrame::Model(ModelFrame::GenerationFinished) => Tag::Finished,
        DataFrame::Model(ModelFrame::ToolCall(c)) => Tag::Call(c.name.to_string()),
        DataFrame::Transcript(Transcript {
            role: Role::Agent,
            finality: Finality::Partial { .. },
            text,
        }) => Tag::Partial(text.to_string()),
        DataFrame::Transcript(Transcript {
            role: Role::Agent,
            finality: Finality::Final,
            text,
        }) => Tag::Final(text.to_string()),
        other => panic!("unexpected frame in generation output: {other:?}"),
    }
}

fn tags(frames: &[DataFrame]) -> Vec<Tag> {
    frames.iter().map(tag).collect()
}

// --- Tests. ------------------------------------------------------------------

#[test]
fn final_user_transcript_streams_append_only_partials_then_final() {
    block_on(async {
        let (mock, _probe) = ScriptedLm::finishing([text("Hel"), text("lo"), text(" there")]);
        let stage = LmStage::new(mock, "you are a test");
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let _ = input.send_data(user_final("hi")).await;
            // Returning drops `input`, cascading shutdown through the pipeline.
        };

        let drain = async move {
            let mut agent = Vec::new();
            while let Some(received) = output.recv().await {
                if let Received::Data(DataFrame::Transcript(t)) = received {
                    if t.role == Role::Agent {
                        agent.push(t);
                    }
                }
            }
            agent
        };

        let (_, agent, _) = futures::join!(feed, drain, driver);

        let (partials, finals): (Vec<_>, Vec<_>) = agent
            .into_iter()
            .partition(|t| matches!(t.finality, Finality::Partial { .. }));

        // Each delta extends the reply, and every partial is fully stable
        // (LM output is append-only: stable == text.len()).
        let texts: Vec<&str> = partials.iter().map(|t| &*t.text).collect();
        assert_eq!(texts, ["Hel", "Hello", "Hello there"]);
        for t in &partials {
            let Finality::Partial { stable } = t.finality else {
                unreachable!()
            };
            assert_eq!(stable, t.text.len(), "an LM partial is fully stable");
        }
        for w in texts.windows(2) {
            assert!(w[1].starts_with(w[0]), "partial {:?} must extend {:?}", w[1], w[0]);
        }

        // Exactly one final, carrying the whole reply.
        assert_eq!(finals.len(), 1, "one final per generation");
        assert_eq!(&*finals[0].text, "Hello there");
    });
}

#[test]
fn text_generation_is_bracketed_by_started_and_finished() {
    block_on(async {
        let (mock, _probe) = ScriptedLm::finishing([text("Sure"), text(" thing.")]);
        let mut stage = LmStage::new(mock, "sys");
        let effect = take_generate(stage.decide_data(&user_final("hi")));
        let frames = run_generation(&stage, effect).await;

        assert_eq!(
            tags(&frames),
            [
                Tag::Started,
                Tag::Partial("Sure".into()),
                Tag::Partial("Sure thing.".into()),
                Tag::Final("Sure thing.".into()),
                Tag::Finished,
            ]
        );
    });
}

#[test]
fn text_then_tool_call_preserves_order() {
    block_on(async {
        let (mock, _probe) = ScriptedLm::finishing([text("Let me check."), call("c1", "lookup")]);
        let mut stage = LmStage::new(mock, "sys");
        let effect = take_generate(stage.decide_data(&user_final("weather?")));
        let frames = run_generation(&stage, effect).await;

        assert_eq!(
            tags(&frames),
            [
                Tag::Started,
                Tag::Partial("Let me check.".into()),
                Tag::Call("lookup".into()),
                Tag::Final("Let me check.".into()),
                Tag::Finished,
            ],
            "the tool call lands between the partial and the final, preserving stream order",
        );
    });
}

#[test]
fn tool_call_then_text_preserves_order() {
    block_on(async {
        let (mock, _probe) = ScriptedLm::finishing([call("c1", "lookup"), text("On it.")]);
        let mut stage = LmStage::new(mock, "sys");
        let effect = take_generate(stage.decide_data(&user_final("weather?")));
        let frames = run_generation(&stage, effect).await;

        // Text after a tool call is not buffered to the end: the partial follows
        // the call.
        assert_eq!(
            tags(&frames),
            [
                Tag::Started,
                Tag::Call("lookup".into()),
                Tag::Partial("On it.".into()),
                Tag::Final("On it.".into()),
                Tag::Finished,
            ]
        );
    });
}

#[test]
fn tool_call_only_response_emits_no_transcript() {
    block_on(async {
        let (mock, _probe) = ScriptedLm::finishing([call("c1", "lookup")]);
        let mut stage = LmStage::new(mock, "sys");
        let effect = take_generate(stage.decide_data(&user_final("weather?")));
        let frames = run_generation(&stage, effect).await;

        assert_eq!(
            tags(&frames),
            [Tag::Started, Tag::Call("lookup".into()), Tag::Finished],
            "a tool-call-only turn emits no empty final transcript",
        );
    });
}

#[test]
fn multiple_tool_calls_each_emit_a_frame() {
    block_on(async {
        let (mock, _probe) = ScriptedLm::finishing([call("c1", "lookup"), call("c2", "book")]);
        let mut stage = LmStage::new(mock, "sys");
        let effect = take_generate(stage.decide_data(&user_final("plan a trip")));
        let frames = run_generation(&stage, effect).await;

        assert_eq!(
            tags(&frames),
            [
                Tag::Started,
                Tag::Call("lookup".into()),
                Tag::Call("book".into()),
                Tag::Finished,
            ]
        );
    });
}

#[test]
fn barge_in_stops_the_reply_within_one_delta_and_cancels() {
    block_on(async {
        let (reached_tx, mut reached_rx) = mpsc::channel::<()>(1);
        let (block_tx, block_rx) = oneshot::channel::<()>();
        let (mock, probe) = ScriptedLm::parking([text("Hel"), text("lo")], reached_tx, block_rx);
        let stage = LmStage::new(mock, "sys");
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
            let _ = input.send_data(user_final("hi")).await;
            // Wait until the first delta is out and the engine has parked, then
            // barge in — the interrupt must land on a parked generation.
            reached_rx
                .next()
                .await
                .expect("the model emits one delta then parks");
            let _ = input
                .send_system(Direction::Down, SystemFrame::Interrupt)
                .await;
            // Returning drops `input`, cascading shutdown through the pipeline.
        };

        let drain = async move {
            let (mut partials, mut finals) = (0usize, 0usize);
            while let Some(received) = output.recv().await {
                if let Received::Data(DataFrame::Transcript(t)) = received {
                    if t.role == Role::Agent {
                        match t.finality {
                            Finality::Partial { .. } => partials += 1,
                            Finality::Final => finals += 1,
                        }
                    }
                }
            }
            (partials, finals)
        };

        let (_, (partials, finals), _) = futures::join!(feed, drain, driver);
        assert_eq!(partials, 1, "only the one delta emitted before the barge-in");
        assert_eq!(finals, 0, "emission stops within one delta: no final");
        assert_eq!(probe.cancels(), 1, "the barge-in must reach the engine's cancel()");
        assert!(
            block_tx.is_canceled(),
            "the in-flight perform must have been dropped"
        );
    });
}

#[test]
fn assistant_turn_is_recorded_across_generations() {
    block_on(async {
        let (mock, probe) = ScriptedLm::finishing([text("sure")]);
        let mut stage = LmStage::new(mock, "sys");

        // Turn 1: the user speaks; the stage generates and records the reply.
        let gen1 = take_generate(stage.decide_data(&user_final("hi")));
        let (out, _in) = link(8);
        stage.perform(gen1, &out).await.unwrap();

        // Turn 2: the conversation this generation sees reveals what was recorded.
        let gen2 = take_generate(stage.decide_data(&user_final("again")));
        let (out2, _in2) = link(8);
        stage.perform(gen2, &out2).await.unwrap();

        let seen = probe.seen();
        assert_eq!(seen.len(), 2, "one generation per user turn");
        assert!(matches!(seen[0].messages[0], Message::System { .. }));
        assert!(matches!(seen[0].messages[1], Message::User { .. }));
        // Turn 1's assistant reply is recorded before turn 2 generates.
        assert!(matches!(
            &seen[1].messages[2],
            Message::Assistant { content, tool_calls }
                if &**content == "sure" && tool_calls.is_empty()
        ));
        assert!(matches!(&seen[1].messages[3], Message::User { content } if &**content == "again"));
    });
}

#[test]
fn tool_calls_survive_in_the_assistant_message() {
    block_on(async {
        let (mock, probe) = ScriptedLm::finishing([text("checking"), call("c1", "lookup")]);
        let mut stage = LmStage::new(mock, "sys");

        let gen1 = take_generate(stage.decide_data(&user_final("weather?")));
        let (out, _in) = link(16);
        stage.perform(gen1, &out).await.unwrap();

        // A second turn reveals the committed assistant message.
        let gen2 = take_generate(stage.decide_data(&user_final("thanks")));
        let (out2, _in2) = link(8);
        stage.perform(gen2, &out2).await.unwrap();

        let seen = probe.seen();
        let Message::Assistant { content, tool_calls } = &seen[1].messages[2] else {
            panic!("turn 1 commits an assistant message");
        };
        assert_eq!(&**content, "checking");
        assert_eq!(tool_calls.len(), 1, "the tool call is preserved in history");
        assert_eq!(&*tool_calls[0].id, "c1");
        assert_eq!(&*tool_calls[0].name, "lookup");
    });
}

#[test]
fn a_barge_in_records_no_assistant_turn() {
    block_on(async {
        let (reached_tx, mut reached_rx) = mpsc::channel::<()>(1);
        // Keep `_block_tx` alive so the parked generation only unparks when dropped.
        let (_block_tx, block_rx) = oneshot::channel::<()>();
        let (mock, probe) = ScriptedLm::parking([text("Hel"), text("lo")], reached_tx, block_rx);
        let mut stage = LmStage::new(mock, "sys");

        // Turn 1: drive the generation until it parks mid-stream, then drop it —
        // exactly what the run loop does to an in-flight `perform` on a barge-in.
        let gen1 = take_generate(stage.decide_data(&user_final("hi")));
        let (out, _in) = link(8);
        {
            let perform = stage.perform(gen1, &out).fuse();
            futures::pin_mut!(perform);
            futures::select! {
                _ = perform => panic!("the parked generation must not complete"),
                _ = reached_rx.next().fuse() => {} // parked: leaving this scope drops `perform`
            }
        }
        // The barge-in also cancels the engine (the run loop's decide_system path).
        stage.decide_system(Direction::Down, &SystemFrame::Interrupt);
        assert_eq!(probe.cancels(), 1, "the barge-in cancels the engine");

        // Turn 2's conversation shows the interrupted reply was never recorded.
        let gen2 = take_generate(stage.decide_data(&user_final("again")));
        let (out2, _in2) = link(8);
        stage.perform(gen2, &out2).await.unwrap();

        let seen = probe.seen();
        let roles: Vec<&str> = seen[1]
            .messages
            .iter()
            .map(|m| match m {
                Message::System { .. } => "system",
                Message::User { .. } => "user",
                Message::Assistant { .. } => "assistant",
                Message::ToolResult { .. } => "tool",
                Message::Event { .. } => "event",
            })
            .collect();
        assert_eq!(
            roles,
            ["system", "user", "user"],
            "an interrupted generation records no assistant turn (documented v1 limitation)",
        );
    });
}

#[test]
fn user_partials_are_consumed_and_non_input_frames_forward() {
    let (mock, _probe) = ScriptedLm::finishing([text("x")]);
    let mut stage = LmStage::new(mock, "sys");

    // A partial user transcript is consumed with no effect — v1 has no speculative
    // prefill yet.
    let partial = stage.decide_data(&Transcript::user_partial("typ", 0).into());
    assert_eq!(partial.disposition, Disposition::Drop);
    assert!(partial.effects.is_empty(), "a user partial emits nothing in v1");

    // A final user transcript is consumed and triggers exactly one generation.
    let final_user = stage.decide_data(&user_final("done"));
    assert_eq!(final_user.disposition, Disposition::Drop);
    assert_eq!(final_user.effects.len(), 1, "a final user turn emits one Generate");

    // An agent transcript (our own output looping back) and any other frame pass
    // through untouched.
    let agent = stage.decide_data(&Transcript::agent_final("hello").into());
    assert_eq!(agent.disposition, Disposition::Forward);
    assert!(agent.effects.is_empty());
}

#[test]
fn model_input_context_appends_without_generating() {
    block_on(async {
        let (mock, probe) = ScriptedLm::finishing([text("ok")]);
        let mut stage = LmStage::new(mock, "sys");

        // A Context tool result is appended but does not generate.
        let ctx = DataFrame::Model(ModelFrame::Input(ModelInput::Context(
            ModelMessage::ToolResult {
                tool_call_id: Arc::from("c1"),
                name: Arc::from("lookup"),
                content: Arc::from("sunny"),
            },
        )));
        let decision = stage.decide_data(&ctx);
        assert_eq!(decision.disposition, Disposition::Drop, "context is consumed");
        assert!(decision.effects.is_empty(), "context does not trigger a generation");

        // A later user turn shows the context is in history ahead of the user turn.
        let effect = take_generate(stage.decide_data(&user_final("and now?")));
        let (out, _in) = link(8);
        stage.perform(effect, &out).await.unwrap();

        let convo = &probe.seen()[0];
        assert!(matches!(
            &convo.messages[1],
            Message::ToolResult { content, .. } if &**content == "sunny"
        ));
        assert!(matches!(&convo.messages[2], Message::User { content } if &**content == "and now?"));
    });
}

#[test]
fn model_input_respond_appends_and_generates() {
    block_on(async {
        let (mock, probe) = ScriptedLm::finishing([text("answering")]);
        let mut stage = LmStage::new(mock, "sys");

        let respond = DataFrame::Model(ModelFrame::Input(ModelInput::Respond(
            ModelMessage::Event {
                source: Arc::from("dispatch"),
                kind: Arc::from("question"),
                content: Arc::from("which seat?"),
            },
        )));
        let effect = take_generate(stage.decide_data(&respond));
        let (out, _in) = link(16);
        stage.perform(effect, &out).await.unwrap();

        let convo = &probe.seen()[0];
        assert!(matches!(
            &convo.messages[1],
            Message::Event { source, kind, content }
                if &**source == "dispatch" && &**kind == "question" && &**content == "which seat?"
        ));
    });
}

#[test]
fn tool_results_survive_conversation_snapshots() {
    block_on(async {
        let (mock, probe) = ScriptedLm::finishing([text("done")]);
        let mut stage = LmStage::new(mock, "sys");

        // A tool call, then its result arriving as context, then a follow-up.
        let _ = stage.decide_data(&user_final("weather?"));
        let ctx = DataFrame::Model(ModelFrame::Input(ModelInput::Context(
            ModelMessage::ToolResult {
                tool_call_id: Arc::from("c1"),
                name: Arc::from("lookup"),
                content: Arc::from("sunny"),
            },
        )));
        let _ = stage.decide_data(&ctx);
        let effect = take_generate(stage.decide_data(&user_final("thanks")));
        let (out, _in) = link(8);
        stage.perform(effect, &out).await.unwrap();

        let convo = &probe.seen()[0];
        let has_tool_result = convo.messages.iter().any(|m| {
            matches!(m, Message::ToolResult { tool_call_id, content, .. }
                if &**tool_call_id == "c1" && &**content == "sunny")
        });
        assert!(has_tool_result, "the tool result persists in the snapshot");
    });
}

// --- Tool configuration. -----------------------------------------------------

#[test]
fn intrinsic_model_tools_are_exposed() {
    let (mock, _probe) = ScriptedLm::with_intrinsic([text("x")], vec![tool("search")]);
    let stage = LmStage::new(mock, "sys");
    let names: Vec<&str> = stage.tools().iter().map(|t| &*t.name).collect();
    assert_eq!(names, ["search"]);
}

#[test]
fn stage_added_tools_are_exposed() {
    let (mock, _probe) = ScriptedLm::finishing([text("x")]);
    let stage = LmStage::with_tools(mock, "sys", [tool("book"), tool("cancel")]).unwrap();
    let names: Vec<&str> = stage.tools().iter().map(|t| &*t.name).collect();
    assert_eq!(names, ["book", "cancel"]);
}

#[test]
fn intrinsic_and_stage_tools_merge() {
    let (mock, _probe) = ScriptedLm::with_intrinsic([text("x")], vec![tool("search")]);
    let stage = LmStage::with_tools(mock, "sys", [tool("book")]).unwrap();
    let names: Vec<&str> = stage.tools().iter().map(|t| &*t.name).collect();
    assert_eq!(names, ["search", "book"], "intrinsic first, then stage-added");
}

#[test]
fn add_tools_extends_the_effective_set() {
    let (mock, _probe) = ScriptedLm::finishing([text("x")]);
    let stage = LmStage::new(mock, "sys")
        .add_tools([tool("a")])
        .unwrap()
        .add_tools([tool("b")])
        .unwrap();
    let names: Vec<&str> = stage.tools().iter().map(|t| &*t.name).collect();
    assert_eq!(names, ["a", "b"]);
}

#[test]
fn duplicate_stage_tool_names_are_rejected() {
    let (mock, _probe) = ScriptedLm::finishing([text("x")]);
    let result = LmStage::with_tools(mock, "sys", [tool("dup"), tool("dup")]);
    assert!(
        matches!(result, Err(LmConfigError::DuplicateToolName { name }) if &*name == "dup"),
        "two stage tools sharing a name are rejected",
    );
}

#[test]
fn a_stage_tool_duplicating_an_intrinsic_tool_is_rejected() {
    let (mock, _probe) = ScriptedLm::with_intrinsic([text("x")], vec![tool("search")]);
    let result = LmStage::with_tools(mock, "sys", [tool("search")]);
    assert!(
        matches!(result, Err(LmConfigError::DuplicateToolName { name }) if &*name == "search"),
        "a stage tool duplicating an intrinsic tool is rejected",
    );
}

#[test]
fn the_effective_tool_set_is_passed_to_generate() {
    block_on(async {
        let (mock, probe) = ScriptedLm::with_intrinsic([text("x")], vec![tool("search")]);
        let mut stage = LmStage::with_tools(mock, "sys", [tool("book")]).unwrap();
        let effect = take_generate(stage.decide_data(&user_final("hi")));
        let (out, _in) = link(8);
        stage.perform(effect, &out).await.unwrap();

        let seen = probe.tools_seen();
        assert_eq!(seen.len(), 1);
        let names: Vec<&str> = seen[0].iter().map(|t| &*t.name).collect();
        assert_eq!(names, ["search", "book"], "every generation receives the merged set");
    });
}
