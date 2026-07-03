//! [`Pipeline`]: a sequence of [`Stage`]s that is itself a [`Stage`].
//!
//! A pipeline *is* a stage ([`impl Stage for Pipeline`](Pipeline#impl-Stage-for-Pipeline)),
//! so pipelines nest: add one to another builder with
//! [`PipelineBuilder::stage`]. It reuses the same [`Inbound`] / [`Outbound`]
//! abstraction every stage connects through — no bespoke channel types in the
//! public surface.
//!
//! # Topology (v1)
//!
//! Both lanes form a linear downstream chain, wired at [`run`](Stage::run)
//! time: the `inbound` handed to the pipeline feeds stage 0, each stage's
//! output feeds the next via [`link`], and the tail's output is the pipeline's
//! `out`. The `sys` lane is threaded through every stage the same way, so a
//! control frame visits each stage in turn, and closing the input cascades
//! shutdown from head to tail.
//!
//! Upstream routing of `Up`-travelling system frames *through* the stages is
//! not yet wired: a [`Stage::perform`] error surfaces on the tail's output,
//! tagged [`Direction::Up`](pipecrab_core::Direction::Up). That is a deliberate
//! v1 limitation.
//!
//! # Driving it
//!
//! [`Pipeline::start`] wires fresh external [`PipelineEnds`] and hands back the
//! driving future. The caller drives it — `block_on` natively, `spawn_local` in
//! the browser; there is no spawning and no executor trait.

use async_trait::async_trait;
use futures::channel::mpsc;

use futures::stream::{FuturesUnordered, StreamExt};
use pipecrab_core::{DataFrame, Decision, Direction, Processor, SystemFrame};

use crate::{Inbound, MaybeSend, Outbound, Stage, StageError};

/// The boxed pipeline driver future returned by [`Pipeline::start`].
///
/// `Send` on native targets, so the driver can be handed to a multi-threaded
/// executor (`tokio::spawn`); `!Send` on `wasm32`, where it is `spawn_local`-ed.
#[cfg(not(target_arch = "wasm32"))]
pub type DriverFuture = futures::future::BoxFuture<'static, ()>;
/// The boxed pipeline driver future returned by [`Pipeline::start`].
#[cfg(target_arch = "wasm32")]
pub type DriverFuture = futures::future::LocalBoxFuture<'static, ()>;

/// Default buffer depth for each lane channel: how many frames may queue on a
/// lane before a send awaits (backpressure). Arbitrary-but-reasonable, not a
/// convention; tune it with [`PipelineBuilder::capacity`].
const DEFAULT_CAPACITY: usize = 16;

/// Create a linked [`Outbound`] / [`Inbound`] pair sharing one data channel and
/// one system channel: frames sent on the `Outbound` are received on the
/// `Inbound`. This is the single wiring primitive — pipelines use it between
/// adjacent stages and at their external ends.
pub fn link(capacity: usize) -> (Outbound, Inbound) {
    let capacity = capacity.max(1);
    let (data_tx, data_rx) = mpsc::channel::<DataFrame>(capacity);
    let (sys_tx, sys_rx) = mpsc::channel::<(Direction, SystemFrame)>(capacity);
    (Outbound { data: data_tx, sys: sys_tx }, Inbound { sys: sys_rx, data: data_rx })
}

/// The external endpoints of a running [`Pipeline`]: send in via `input`, read
/// out via `output` — the same [`Outbound`] / [`Inbound`] every stage uses.
///
/// Returned by [`Pipeline::start`]. Dropping `input` closes the head's inbound
/// lanes, which cascades shutdown downstream and lets the driver finish.
pub struct PipelineEnds {
    /// Send data and system frames into the head of the pipeline.
    pub input: Outbound,
    /// Receive data and system frames emitted past the tail of the pipeline.
    pub output: Inbound,
}

/// Builds a [`Pipeline`] from an ordered list of stages sharing one `Effect`.
pub struct PipelineBuilder<E> {
    stages: Vec<Box<dyn Stage<Effect = E>>>,
    capacity: usize,
}

impl<E: MaybeSend + 'static> PipelineBuilder<E> {
    /// A new, empty builder with the default lane capacity.
    pub fn new() -> Self {
        Self { stages: Vec::new(), capacity: DEFAULT_CAPACITY }
    }

    /// Override the per-lane buffer depth — how many frames may queue on a lane
    /// before a send awaits (backpressure). Clamped to at least 1.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(1);
        self
    }

    /// Append a stage, boxing it. Stages run in the order added, head first. A
    /// [`Pipeline`] is itself a stage, so it may be passed here to nest.
    pub fn stage(mut self, stage: impl Stage<Effect = E> + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Append an already-boxed stage.
    pub fn boxed(mut self, stage: Box<dyn Stage<Effect = E>>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Finish building.
    ///
    /// # Panics
    ///
    /// Panics if no stages were added — a pipeline needs at least one stage.
    pub fn build(self) -> Pipeline<E> {
        assert!(!self.stages.is_empty(), "a pipeline needs at least one stage");
        Pipeline { stages: self.stages, capacity: self.capacity }
    }
}

impl<E: MaybeSend + 'static> Default for PipelineBuilder<E> {
    fn default() -> Self {
        Self::new()
    }
}

/// A sequence of stages, itself a [`Stage`]. Wired and driven by [`start`] (at
/// the top level) or by a parent pipeline's [`run`](Stage::run) (when nested).
///
/// [`start`]: Pipeline::start
pub struct Pipeline<E> {
    stages: Vec<Box<dyn Stage<Effect = E>>>,
    capacity: usize,
}

impl<E: MaybeSend + 'static> Pipeline<E> {
    /// Wire fresh external [`PipelineEnds`] and return them with the driving
    /// future. The caller drives the future (e.g. `block_on`) and uses the ends
    /// to feed the head and read the tail.
    pub fn start(self) -> (PipelineEnds, DriverFuture) {
        let capacity = self.capacity;
        let (input, head_in) = link(capacity);
        let (tail_out, output) = link(capacity);
        let driver = Box::new(self).run(head_in, tail_out);
        (PipelineEnds { input, output }, driver)
    }
}

// A pipeline's behavior lives entirely in its overridden `run`, which drives its
// children. It is never treated as a leaf, so `decide_*` and `perform` are never
// reached in correct use — they panic as tripwires for misuse (e.g. a custom
// driver that calls the wrong method). `run` is the one legitimate entry point.
impl<E> Processor for Pipeline<E> {
    type Effect = E;

    fn decide_data(&mut self, _frame: &DataFrame) -> Decision<E> {
        unreachable!("a Pipeline is driven by Stage::run, not decide_data")
    }

    fn decide_system(&mut self, _dir: Direction, _frame: &SystemFrame) -> Decision<E> {
        unreachable!("a Pipeline is driven by Stage::run, not decide_system")
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<E: MaybeSend + 'static> Stage for Pipeline<E> {
    async fn perform(&self, _effect: E, _out: &Outbound) -> Result<(), StageError> {
        unreachable!("a Pipeline is driven by Stage::run, not perform")
    }

    /// Wire the children between `inbound` and `out` and drive every child's
    /// `run` cooperatively as one future via [`FuturesUnordered`].
    async fn run(self: Box<Self>, inbound: Inbound, out: Outbound) {
        let Pipeline { stages, capacity } = *self;
        let n = stages.len();
        let mut tasks = FuturesUnordered::new();

        // Thread `inbound` into stage 0 and `out` out of the last stage; link
        // adjacent stages with fresh channels in between.
        let mut current_in = Some(inbound);
        let mut final_out = Some(out);
        for (i, stage) in stages.into_iter().enumerate() {
            let stage_in = current_in.take().expect("inbound threaded through");
            let stage_out = if i + 1 == n {
                final_out.take().expect("outbound threaded through")
            } else {
                let (this_out, next_in) = link(capacity);
                current_in = Some(next_in);
                this_out
            };
            tasks.push(stage.run(stage_in, stage_out));
        }

        while tasks.next().await.is_some() {}
    }
}
