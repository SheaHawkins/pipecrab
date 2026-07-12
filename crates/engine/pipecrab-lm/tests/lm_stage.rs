//! `LmStage` adapts a `LanguageModel` into a stage: on a final user `Transcript`
//! it appends the turn to the running conversation and streams a generated reply
//! back as agent transcripts — append-only partials, then a final — recording the
//! assistant turn only once it completes.
//!
//! The append-only streaming and barge-in integration go through the real
//! pipeline; the conversation-recording invariants are pinned at the `decide`/
//! `perform` level, where the (private) conversation is observed indirectly via
//! the scripted mock's record of what each later generation was asked to generate
//! from.
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
    DataFrame, Decision, Direction, Disposition, Finality, Processor, Role, SystemFrame, Transcript,
};
use pipecrab_lm::{
    ChatRole, Conversation, GenParams, Generate, LanguageModel, LmError, LmStage, TokenOut,
    TokenStream,
};
use pipecrab_runtime::{link, PipelineBuilder, Received, Stage};

// --- A scripted LanguageModel mock. ------------------------------------------

/// What a [`ScriptedLm`] observed, shared with the test via an `Arc`.
#[derive(Default)]
struct LmProbe {
    cancels: AtomicUsize,
    seen: Mutex<Vec<Conversation>>,
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
}

/// One `unfold` step of a parked generation stream.
enum Step {
    /// Emit this delta, then advance to [`Step::Park`].
    Emit(Arc<str>, mpsc::Sender<()>, oneshot::Receiver<()>),
    /// Signal the test that the stream has parked, then wait to be released — a
    /// barge-in drops this future (cancelling `block`) before it resolves.
    Park(mpsc::Sender<()>, oneshot::Receiver<()>),
}

/// A hardware-free [`LanguageModel`]: yields scripted deltas, records every call,
/// and — when built with [`parking`](ScriptedLm::parking) — parks its first
/// generation after one delta so a barge-in can drop it in flight.
struct ScriptedLm {
    deltas: Vec<Arc<str>>,
    /// `Some` until the first (parking) generation consumes it; later generations
    /// finish normally.
    park: Mutex<Option<(mpsc::Sender<()>, oneshot::Receiver<()>)>>,
    probe: Arc<LmProbe>,
}

impl ScriptedLm {
    /// A model that yields `deltas` and completes on every generation.
    fn finishing<I, S>(deltas: I) -> (Self, Arc<LmProbe>)
    where
        I: IntoIterator<Item = S>,
        S: Into<Arc<str>>,
    {
        let probe = Arc::new(LmProbe::default());
        let me = Self {
            deltas: deltas.into_iter().map(Into::into).collect(),
            park: Mutex::new(None),
            probe: probe.clone(),
        };
        (me, probe)
    }

    /// A model whose **first** generation yields one delta, signals on `reached`,
    /// then parks on `block` (a barge-in drops the future here); every later
    /// generation finishes normally.
    fn parking<I, S>(
        deltas: I,
        reached: mpsc::Sender<()>,
        block: oneshot::Receiver<()>,
    ) -> (Self, Arc<LmProbe>)
    where
        I: IntoIterator<Item = S>,
        S: Into<Arc<str>>,
    {
        let probe = Arc::new(LmProbe::default());
        let me = Self {
            deltas: deltas.into_iter().map(Into::into).collect(),
            park: Mutex::new(Some((reached, block))),
            probe: probe.clone(),
        };
        (me, probe)
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl LanguageModel for ScriptedLm {
    async fn generate(
        &self,
        convo: &Conversation,
        _params: &GenParams,
    ) -> Result<TokenStream, LmError> {
        self.probe.seen.lock().unwrap().push(convo.clone());
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
                        Some((Ok(TokenOut { delta }), Step::Park(reached, block)))
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
            let items: Vec<Result<TokenOut, LmError>> = deltas
                .into_iter()
                .map(|delta| Ok(TokenOut { delta }))
                .collect();
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

/// A final user transcript as a data frame — the input that drives a generation.
fn user_final(text: &str) -> DataFrame {
    Transcript::user_final(text).into()
}

/// Extract the single `Generate` a final user turn must emit, asserting it is
/// consumed rather than forwarded.
fn take_generate(d: Decision<Generate>) -> Generate {
    assert_eq!(
        d.disposition,
        Disposition::Drop,
        "a final user turn is consumed"
    );
    d.effects
        .into_iter()
        .next()
        .expect("a final user turn emits one Generate")
}

/// The role sequence of a recorded conversation.
fn roles(c: &Conversation) -> Vec<ChatRole> {
    c.messages.iter().map(|m| m.role).collect()
}

// --- Tests. ------------------------------------------------------------------

#[test]
fn final_user_transcript_streams_append_only_partials_then_final() {
    block_on(async {
        let (mock, _probe) = ScriptedLm::finishing(["Hel", "lo", " there"]);
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
            assert!(
                w[1].starts_with(w[0]),
                "partial {:?} must extend {:?}",
                w[1],
                w[0]
            );
        }

        // Exactly one final, carrying the whole reply.
        assert_eq!(finals.len(), 1, "one final per generation");
        assert_eq!(&*finals[0].text, "Hello there");
    });
}

#[test]
fn barge_in_stops_the_reply_within_one_delta_and_cancels() {
    block_on(async {
        let (reached_tx, mut reached_rx) = mpsc::channel::<()>(1);
        let (block_tx, block_rx) = oneshot::channel::<()>();
        let (mock, probe) = ScriptedLm::parking(["Hel", "lo"], reached_tx, block_rx);
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
        assert_eq!(
            partials, 1,
            "only the one delta emitted before the barge-in"
        );
        assert_eq!(finals, 0, "emission stops within one delta: no final");
        assert_eq!(
            probe.cancels(),
            1,
            "the barge-in must reach the engine's cancel()"
        );
        assert!(
            block_tx.is_canceled(),
            "the in-flight perform must have been dropped"
        );
    });
}

#[test]
fn assistant_turn_is_recorded_across_generations() {
    block_on(async {
        let (mock, probe) = ScriptedLm::finishing(["sure"]);
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
        assert_eq!(roles(&seen[0]), [ChatRole::System, ChatRole::User]);
        assert_eq!(
            roles(&seen[1]),
            [
                ChatRole::System,
                ChatRole::User,
                ChatRole::Assistant,
                ChatRole::User
            ],
            "turn 1's reply is recorded before turn 2 generates",
        );
        assert_eq!(&*seen[1].messages[2].content, "sure");
    });
}

#[test]
fn a_barge_in_records_no_assistant_turn() {
    block_on(async {
        let (reached_tx, mut reached_rx) = mpsc::channel::<()>(1);
        // Keep `_block_tx` alive so the parked generation only unparks when dropped.
        let (_block_tx, block_rx) = oneshot::channel::<()>();
        let (mock, probe) = ScriptedLm::parking(["Hel", "lo"], reached_tx, block_rx);
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
        assert_eq!(
            roles(&seen[1]),
            [ChatRole::System, ChatRole::User, ChatRole::User],
            "an interrupted generation records no assistant turn (documented v1 limitation)",
        );
    });
}

#[test]
fn user_partials_are_consumed_and_non_user_frames_forward() {
    let (mock, _probe) = ScriptedLm::finishing(["x"]);
    let mut stage = LmStage::new(mock, "sys");

    // A partial user transcript is consumed with no effect — v1 has no speculative
    // prefill yet.
    let partial = stage.decide_data(&Transcript::user_partial("typ", 0).into());
    assert_eq!(partial.disposition, Disposition::Drop);
    assert!(
        partial.effects.is_empty(),
        "a user partial emits nothing in v1"
    );

    // A final user transcript is consumed and triggers exactly one generation.
    let final_user = stage.decide_data(&user_final("done"));
    assert_eq!(final_user.disposition, Disposition::Drop);
    assert_eq!(
        final_user.effects.len(),
        1,
        "a final user turn emits one Generate"
    );

    // An agent transcript (our own output looping back) and any other frame pass
    // through untouched.
    let agent = stage.decide_data(&Transcript::agent_final("hello").into());
    assert_eq!(agent.disposition, Disposition::Forward);
    assert!(agent.effects.is_empty());
}
