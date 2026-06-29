//! Run-loop behavior: interrupt barge-in, sys-preempts-data, pass-through.
//!
//! All deterministic and tokio-free, driven by `futures::executor::block_on`.
//! The interrupt test parks `perform` on a `oneshot` the test never fires, so
//! the only way `run()` can terminate is by abandoning that `perform` — the
//! test hanging would itself be the failure signal; the assertions confirm the
//! mechanism (the receiver was dropped, and `decide_system(Interrupt)` ran).

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::executor::block_on;
use futures::future::join;
use futures::sink::SinkExt;
use futures::stream::StreamExt;
use pipecrab_core::{DataFrame, Decision, Direction, Processor, SystemFrame};
use pipecrab_runtime::{Outbound, PipelineBuilder, PipelineEnds, Stage, StageError};

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
        let (ends, pipeline) = PipelineBuilder::new().stage(stage).build();
        let PipelineEnds { mut data_in, mut sys_in, .. } = ends;

        let feeder = async move {
            data_in.send(DataFrame::Transcript("go".into())).await.unwrap();
            started_rx.next().await.expect("perform must start");
            sys_in.send((Direction::Down, SystemFrame::Interrupt)).await.unwrap();
            // Returning drops data_in & sys_in -> inbound closes -> the loop exits.
        };

        join(feeder, pipeline.run()).await;

        assert!(interrupted.get(), "decide_system(Interrupt) must have run");
        assert!(
            block_tx.is_canceled(),
            "the in-flight perform must have been dropped (its receiver gone)",
        );
    });
}

// --- Test 2: a system frame preempts a backed-up data lane.

/// Counts data frames in `decide_data`; on Interrupt, records how many had been
/// processed at that moment.
struct CountingStage {
    data_count: Rc<Cell<usize>>,
    data_at_interrupt: Rc<Cell<Option<usize>>>,
}

impl Processor for CountingStage {
    type Effect = ();
    fn decide_data(&mut self, _frame: &DataFrame) -> Decision<()> {
        self.data_count.set(self.data_count.get() + 1);
        Decision::drop() // no effect -> perform is never called
    }
    fn decide_system(&mut self, _dir: Direction, frame: &SystemFrame) -> Decision<()> {
        if matches!(frame, SystemFrame::Interrupt) {
            self.data_at_interrupt.set(Some(self.data_count.get()));
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
        let data_at_interrupt = Rc::new(Cell::new(None));
        let stage = CountingStage {
            data_count: data_count.clone(),
            data_at_interrupt: data_at_interrupt.clone(),
        };
        let (ends, pipeline) = PipelineBuilder::new().stage(stage).build();
        let PipelineEnds { mut data_in, mut sys_in, .. } = ends;

        // Back up the data lane, then enqueue an Interrupt behind it.
        for i in 0..8 {
            data_in.send(DataFrame::Transcript(i.to_string().into())).await.unwrap();
        }
        sys_in.send((Direction::Down, SystemFrame::Interrupt)).await.unwrap();
        drop(data_in);
        drop(sys_in);

        pipeline.run().await;

        assert_eq!(
            data_at_interrupt.get(),
            Some(0),
            "the Interrupt must jump the 8-frame data backlog",
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
        let (ends, pipeline) = PipelineBuilder::new().stage(PassThrough).build();
        let PipelineEnds { data_in, sys_in, mut data_out, sys_out: _sys_out } = ends;

        let feeder = async move {
            let mut data_in = data_in;
            let _sys_in = sys_in; // dropped with data_in at block end -> shutdown
            data_in.send(DataFrame::Transcript("hi".into())).await.unwrap();
        };

        join(feeder, pipeline.run()).await;

        match data_out.next().await {
            Some(DataFrame::Transcript(s)) => assert_eq!(&*s, "hi"),
            other => panic!("expected forwarded Transcript(hi), got {other:?}"),
        }
    });
}
