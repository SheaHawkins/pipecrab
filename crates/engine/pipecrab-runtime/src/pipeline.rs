//! A sequence of [`Stage`]s that is itself a stage.
//!
//! # Topology (v1)
//!
//! Both lanes form a linear downstream chain wired by [`Stage::run`].
//!
//! Upstream system frames are exposed at the tail rather than routed back
//! through preceding stages.
//!
//! # Driving it
//!
//! [`Pipeline::start`] returns endpoints and a future for the caller to drive.

use async_trait::async_trait;
use futures::channel::mpsc;

use futures::stream::{FuturesUnordered, StreamExt};
use pipecrab_core::{DataFrame, Decision, Direction, Processor, SystemFrame};

use crate::{Inbound, MaybeSend, MaybeSendSync, Outbound, Stage, StageError};

/// The boxed driver future returned by [`Pipeline::start`].
#[cfg(not(target_arch = "wasm32"))]
pub type DriverFuture = futures::future::BoxFuture<'static, ()>;
/// The boxed pipeline driver future returned by [`Pipeline::start`].
#[cfg(target_arch = "wasm32")]
pub type DriverFuture = futures::future::LocalBoxFuture<'static, ()>;

/// Default capacity of each lane.
const DEFAULT_CAPACITY: usize = 16;

/// Creates linked [`Outbound`] and [`Inbound`] endpoints with two typed lanes.
pub fn link(capacity: usize) -> (Outbound, Inbound) {
    let capacity = capacity.max(1);
    let (data_tx, data_rx) = mpsc::channel::<DataFrame>(capacity);
    let (sys_tx, sys_rx) = mpsc::channel::<(Direction, SystemFrame)>(capacity);
    (
        Outbound {
            data: data_tx,
            sys: sys_tx,
        },
        Inbound {
            sys: sys_rx,
            data: data_rx,
        },
    )
}

/// The external endpoints of a running [`Pipeline`].
///
/// Dropping `input` closes the pipeline and allows its driver to finish.
pub struct PipelineEnds {
    /// Send data and system frames into the head of the pipeline.
    pub input: Outbound,
    /// Receive data and system frames emitted past the tail of the pipeline.
    pub output: Inbound,
}

/// Type-erased runner for a stored stage.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
trait ErasedStage: MaybeSendSync {
    async fn run(self: Box<Self>, inbound: Inbound, out: Outbound);
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<S> ErasedStage for S
where
    S: Stage + 'static,
    S::Effect: MaybeSend,
{
    async fn run(self: Box<Self>, inbound: Inbound, out: Outbound) {
        Stage::run(self, inbound, out).await;
    }
}

/// Adapts an already boxed stage to the pipeline's erased runner.
struct BoxedStage<E: MaybeSend>(Box<dyn Stage<Effect = E>>);

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<E: MaybeSend + 'static> ErasedStage for BoxedStage<E> {
    async fn run(self: Box<Self>, inbound: Inbound, out: Outbound) {
        self.0.run(inbound, out).await;
    }
}

/// Builds a [`Pipeline`] from an ordered list of stages.
pub struct PipelineBuilder {
    stages: Vec<Box<dyn ErasedStage>>,
    capacity: usize,
}

impl PipelineBuilder {
    /// A new, empty builder with the default lane capacity.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            capacity: DEFAULT_CAPACITY,
        }
    }

    /// Sets each lane's capacity, clamped to at least one.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(1);
        self
    }

    /// Appends a stage. Stages run in insertion order.
    pub fn stage<S>(mut self, stage: S) -> Self
    where
        S: Stage + 'static,
        S::Effect: MaybeSend,
    {
        self.stages.push(Box::new(stage));
        self
    }

    /// Append an already-boxed stage.
    pub fn boxed<E: MaybeSend + 'static>(mut self, stage: Box<dyn Stage<Effect = E>>) -> Self {
        self.stages.push(Box::new(BoxedStage(stage)));
        self
    }

    /// Finish building.
    ///
    /// # Panics
    ///
    /// Panics if no stages were added — a pipeline needs at least one stage.
    pub fn build(self) -> Pipeline {
        assert!(
            !self.stages.is_empty(),
            "a pipeline needs at least one stage"
        );
        Pipeline {
            stages: self.stages,
            capacity: self.capacity,
        }
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A sequence of stages that implements [`Stage`].
///
/// [`start`]: Pipeline::start
pub struct Pipeline {
    stages: Vec<Box<dyn ErasedStage>>,
    capacity: usize,
}

impl Pipeline {
    /// Returns fresh endpoints and the future that drives the pipeline.
    pub fn start(self) -> (PipelineEnds, DriverFuture) {
        let capacity = self.capacity;
        let (input, head_in) = link(capacity);
        let (tail_out, output) = link(capacity);
        let driver = Stage::run(Box::new(self), head_in, tail_out);
        (PipelineEnds { input, output }, driver)
    }
}

// A pipeline's behavior lives entirely in its overridden `run`, which drives its
// children. It is never treated as a leaf, so `decide_*` and `perform` are never
// reached in correct use — they panic as tripwires for misuse (e.g. a custom
// driver that calls the wrong method). `run` is the one legitimate entry point.
impl Processor for Pipeline {
    type Effect = ();

    fn decide_data(&mut self, _frame: &DataFrame) -> Decision<()> {
        unreachable!("a Pipeline is driven by Stage::run, not decide_data")
    }

    fn decide_system(&mut self, _dir: Direction, _frame: &SystemFrame) -> Decision<()> {
        unreachable!("a Pipeline is driven by Stage::run, not decide_system")
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Stage for Pipeline {
    async fn perform(&self, _effect: (), _out: &Outbound) -> Result<(), StageError> {
        unreachable!("a Pipeline is driven by Stage::run, not perform")
    }

    /// Wires and drives every child stage.
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
            tasks.push(ErasedStage::run(stage, stage_in, stage_out));
        }

        while tasks.next().await.is_some() {}
    }
}
