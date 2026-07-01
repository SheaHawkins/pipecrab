//! Run-loop behavior: interrupt barge-in, sys-preempts-data, pass-through.
//!
//! All deterministic and tokio-free, driven by `futures::executor::block_on`.
//! Frames go in through the pipeline's `input` ([`Outbound`]) and come out
//! through its `output` ([`Inbound`]) — the same abstraction every stage uses.
//!
//! The interrupt test parks `perform` on a `oneshot` the test never fires, so
//! the only way the driver can terminate is by abandoning that `perform` — the
//! test hanging would itself be the failure signal; the assertions confirm the
//! mechanism (the receiver was dropped, and `decide_system(Interrupt)` ran).

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::executor::block_on;
use futures::future::join;
use futures::sink::SinkExt;
use futures::stream::StreamExt;
use pipecrab_core::{AudioChunk, AudioFormat, DataFrame, Decision, Direction, Processor, SystemFrame};
use pipecrab_runtime::{Outbound, PipelineBuilder, Received, Stage, StageError};

// --- Test 1: an Interrupt abandons an in-flight perform and runs decide_system.

/// `perform` signals that it started, then parks forever on a `oneshot` the
/// test never fires. `decide_system(Interrupt)` flips the shared flag.
struct BlockingStage {
    block_rx: RefCell<Option<oneshot::Receiver<()>>>,
    started: mpsc::Sender<()>,
    interrupted: Rc<Cell<bool>>,
}

impl Processor for BlockingStage {
    type Effect = ();
    fn decide_data(&mut self, _frame: &DataFrame) -> Decision<()> {
        Decision::drop().emit(()) // drop the input; emit one effect to perform
    }
    fn decide_system(&mut self, _dir: Direction, frame: &SystemFrame) -> Decision<()> {
        if matches!(frame, SystemFrame::Interrupt) {
            self.interrupted.set(true);
        }
        Decision::drop()
    }
}

#[async_trait(?Send)]
impl Stage for BlockingStage {
    async fn perform(&self, _effect: (), _out: &Outbound) -> Result<(), StageError> {
        let _ = self.started.clone().send(()).await;
        let rx = self.block_rx.borrow_mut().take().expect("perform runs once");
        let _ = rx.await; // never fires; the receiver drops when perform is abandoned
        Ok(())
    }
}

#[test]
fn interrupt_abandons_perform_and_runs_decide_system() {
    block_on(async {
        let interrupted = Rc::new(Cell::new(false));
        let (started_tx, mut started_rx) = mpsc::channel::<()>(1);
        let (block_tx, block_rx) = oneshot::channel::<()>();

        let stage = BlockingStage {
            block_rx: RefCell::new(Some(block_rx)),
            started: started_tx,
            interrupted: interrupted.clone(),
        };
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input; // Outbound: send into the pipeline head
        let _output = ends.output; // keep the tail's output channel open

        let feeder = async move {
            input.send_data(DataFrame::Transcript("go".into())).await.unwrap();
            started_rx.next().await.expect("perform must start");
            input.send_system(Direction::Down, SystemFrame::Interrupt).await.unwrap();
            // Returning drops `input` -> head inbound closes -> the driver exits.
        };

        join(feeder, driver).await;

        assert!(interrupted.get(), "decide_system(Interrupt) must have run");
        assert!(
            block_tx.is_canceled(),
            "the in-flight perform must have been dropped (its receiver gone)",
        );
    });
}

// --- Test 2: a system frame preempts a backed-up data lane.

/// Counts data frames in `decide_data`; on a `Start` frame, records how many had
/// been processed at that moment.
struct CountingStage {
    data_count: Rc<Cell<usize>>,
    data_at_preempt: Rc<Cell<Option<usize>>>,
}

impl Processor for CountingStage {
    type Effect = ();
    fn decide_data(&mut self, _frame: &DataFrame) -> Decision<()> {
        self.data_count.set(self.data_count.get() + 1);
        Decision::drop() // no effect -> perform is never called
    }
    fn decide_system(&mut self, _dir: Direction, frame: &SystemFrame) -> Decision<()> {
        // Record on `Start`, a frame that preempts but does *not* flush — this
        // isolates lane preemption from the interrupt data-flush (tested below).
        if matches!(frame, SystemFrame::Start) {
            self.data_at_preempt.set(Some(self.data_count.get()));
        }
        Decision::drop()
    }
}

#[async_trait(?Send)]
impl Stage for CountingStage {
    async fn perform(&self, _effect: (), _out: &Outbound) -> Result<(), StageError> {
        Ok(())
    }
}

#[test]
fn sys_preempts_backed_up_data() {
    block_on(async {
        let data_count = Rc::new(Cell::new(0usize));
        let data_at_preempt = Rc::new(Cell::new(None));
        let stage = CountingStage {
            data_count: data_count.clone(),
            data_at_preempt: data_at_preempt.clone(),
        };
        let (ends, driver) = PipelineBuilder::new().stage(stage).build().start();
        let input = ends.input;
        let _output = ends.output;

        // Back up the data lane, then enqueue a (non-flushing) Start behind it.
        for i in 0..8 {
            input.send_data(DataFrame::Transcript(i.to_string().into())).await.unwrap();
        }
        input.send_system(Direction::Down, SystemFrame::Start).await.unwrap();
        drop(input);

        driver.await;

        assert_eq!(
            data_at_preempt.get(),
            Some(0),
            "the Start frame must jump the 8-frame data backlog",
        );
        assert_eq!(data_count.get(), 8, "all backed-up data is still processed afterward");
    });
}

// --- Test 3: an un-overridden stage is a transparent pass-through.

/// Every `Processor`/`Stage` method left at its default.
struct PassThrough;

impl Processor for PassThrough {
    type Effect = ();
}

#[async_trait(?Send)]
impl Stage for PassThrough {
    async fn perform(&self, _effect: (), _out: &Outbound) -> Result<(), StageError> {
        Ok(())
    }
}

#[test]
fn pass_through_forwards_data() {
    block_on(async {
        let (ends, driver) = PipelineBuilder::new().stage(PassThrough).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feeder = async move {
            input.send_data(DataFrame::Transcript("hi".into())).await.unwrap();
            // Dropping `input` at block end closes the head -> shutdown.
        };

        join(feeder, driver).await;

        match output.recv().await {
            Some(Received::Data(DataFrame::Transcript(s))) => assert_eq!(&*s, "hi"),
            other => panic!("expected forwarded Transcript(hi), got {other:?}"),
        }
    });
}

// --- Test 4: an Interrupt flushes the data backlog, keeping transport-audio.

fn input_audio(id: u8) -> DataFrame {
    DataFrame::InputAudio { bytes: Arc::from(&[id][..]), sample_rate: 16_000, num_channels: 1 }
}

#[test]
fn interrupt_flushes_data_keeping_survivors_in_order() {
    block_on(async {
        // PassThrough forwards everything, so without the flush all four data
        // frames would reach `output`; with it, only the two InputAudio survive.
        let (ends, driver) = PipelineBuilder::new().stage(PassThrough).build().start();
        let input = ends.input;
        let mut output = ends.output;

        // Back up the data lane with survivors interleaved with droppable frames,
        // then an Interrupt behind it — sys-biased recv handles it first, while
        // the whole backlog is still queued.
        input.send_data(input_audio(1)).await.unwrap();
        input.send_data(DataFrame::Transcript("drop me".into())).await.unwrap();
        input.send_data(input_audio(2)).await.unwrap();
        let audio = AudioChunk::new(Arc::from(&[0.0f32, 0.0][..]), AudioFormat::new(48_000, 1));
        input.send_data(DataFrame::Audio(audio)).await.unwrap();
        input.send_system(Direction::Down, SystemFrame::Interrupt).await.unwrap();
        drop(input);

        driver.await;

        // Drain the data lane: only the two InputAudio frames, in arrival order.
        let mut ids = Vec::new();
        while let Ok(frame) = output.data.try_recv() {
            match frame {
                DataFrame::InputAudio { bytes, .. } => ids.push(bytes[0]),
                other => panic!("a non-survivor leaked past the flush: {other:?}"),
            }
        }
        assert_eq!(ids, vec![1, 2], "survivors kept in order; droppable frames flushed");
    });
}

// --- Test 5: a pipeline is a stage, so pipelines nest.

#[test]
fn nested_pipeline_forwards_through_both_levels() {
    block_on(async {
        // Inner pipeline is a single pass-through; nest it inside an outer one
        // that also has a pass-through. A frame must traverse both levels.
        let inner = PipelineBuilder::new().stage(PassThrough).build();
        let (ends, driver) =
            PipelineBuilder::new().stage(inner).stage(PassThrough).build().start();
        let input = ends.input;
        let mut output = ends.output;

        let feeder = async move {
            input.send_data(DataFrame::Transcript("deep".into())).await.unwrap();
        };

        join(feeder, driver).await;

        match output.recv().await {
            Some(Received::Data(DataFrame::Transcript(s))) => assert_eq!(&*s, "deep"),
            other => panic!("expected forwarded Transcript(deep), got {other:?}"),
        }
    });
}
