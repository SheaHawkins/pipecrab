//! [`StageObserver`] behavior: wiring labels, per-stage callback ordering,
//! tail-tap visibility, nested-pipeline propagation, and the aborted-perform
//! pair on a barge-in.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::executor::block_on;
use futures::future::join;
use futures::sink::SinkExt;
use futures::stream::StreamExt;
use pipecrab_core::{
    DataFrame, Decision, Direction, Disposition, Processor, SystemFrame, Transcript,
};
use pipecrab_runtime::{
    Outbound, PerformOutcome, PipelineBuilder, Stage, StageError, StageId, StageObserver,
    TAIL_STAGE,
};

/// One observer callback, flattened for assertion.
#[derive(Debug, Clone, PartialEq)]
enum Seen {
    Wired(Vec<(String, &'static str)>),
    DataIn(String, &'static str),
    DataDecided(String, Disposition, usize),
    SysIn(String, &'static str),
    SysDecided(String, Disposition, usize),
    PerformStart(String),
    PerformEnd(String, PerformOutcome),
}

#[derive(Default)]
struct Recorder(Mutex<Vec<Seen>>);

impl Recorder {
    fn take(&self) -> Vec<Seen> {
        std::mem::take(&mut self.0.lock().unwrap())
    }
    fn push(&self, seen: Seen) {
        self.0.lock().unwrap().push(seen);
    }
}

impl StageObserver for Recorder {
    fn wired(&self, stages: &[StageId]) {
        self.push(Seen::Wired(
            stages
                .iter()
                .map(|s| (s.path.to_string(), s.name))
                .collect(),
        ));
    }
    fn data_in(&self, stage: &StageId, _frame: &DataFrame) {
        self.push(Seen::DataIn(stage.path.to_string(), stage.name));
    }
    fn data_decided(&self, stage: &StageId, disposition: Disposition, effects: usize) {
        self.push(Seen::DataDecided(
            stage.path.to_string(),
            disposition,
            effects,
        ));
    }
    fn sys_in(&self, stage: &StageId, _dir: Direction, frame: &SystemFrame) {
        let kind = match frame {
            SystemFrame::Start => "Start",
            SystemFrame::Stop => "Stop",
            SystemFrame::Interrupt => "Interrupt",
            SystemFrame::Error { .. } => "Error",
        };
        self.push(Seen::SysIn(stage.path.to_string(), kind));
    }
    fn sys_decided(&self, stage: &StageId, disposition: Disposition, effects: usize) {
        self.push(Seen::SysDecided(
            stage.path.to_string(),
            disposition,
            effects,
        ));
    }
    fn perform_start(&self, stage: &StageId) {
        self.push(Seen::PerformStart(stage.path.to_string()));
    }
    fn perform_end(&self, stage: &StageId, outcome: PerformOutcome) {
        self.push(Seen::PerformEnd(stage.path.to_string(), outcome));
    }
}

/// Replaces each user-final transcript with an upper-cased one (drop + emit).
struct Shout;

struct ShoutEffect(Arc<str>);

impl Processor for Shout {
    type Effect = ShoutEffect;
    fn name(&self) -> &'static str {
        "shout"
    }
    fn decide_data(&mut self, frame: &DataFrame) -> Decision<ShoutEffect> {
        match frame {
            DataFrame::Transcript(t) => Decision::drop().emit(ShoutEffect(t.text.clone())),
            _ => Decision::forward(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Stage for Shout {
    async fn perform(&self, effect: ShoutEffect, out: &Outbound) -> Result<(), StageError> {
        let _ = out
            .send_data(Transcript::user_final(effect.0.to_uppercase()).into())
            .await;
        Ok(())
    }
}

struct PassThrough;
impl Processor for PassThrough {
    type Effect = ();
    fn name(&self) -> &'static str {
        "pass"
    }
}
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Stage for PassThrough {
    async fn perform(&self, _effect: (), _out: &Outbound) -> Result<(), StageError> {
        Ok(())
    }
}

#[test]
fn observer_sees_wiring_decides_performs_and_tail() {
    block_on(async {
        let recorder = Arc::new(Recorder::default());
        let (ends, driver) = PipelineBuilder::new()
            .stage(Shout)
            .observer(recorder.clone())
            .build()
            .start();
        let input = ends.input;
        let _output = ends.output;

        let feeder = async move {
            input
                .send_data(Transcript::user_final("hi").into())
                .await
                .unwrap();
        };
        join(feeder, driver).await;

        let seen = recorder.take();
        assert_eq!(
            seen,
            vec![
                // Wiring: the user stage plus the appended tail tap.
                Seen::Wired(vec![("0".into(), "shout"), ("1".into(), TAIL_STAGE)]),
                // The transcript reaches shout, is dropped, one effect queued.
                Seen::DataIn("0".into(), "shout"),
                Seen::DataDecided("0".into(), Disposition::Drop, 1),
                Seen::PerformStart("0".into()),
                Seen::PerformEnd("0".into(), PerformOutcome::Ok),
                // The emitted replacement crosses the tail tap on its way out.
                Seen::DataIn("1".into(), TAIL_STAGE),
                Seen::DataDecided("1".into(), Disposition::Forward, 0),
            ],
        );
    });
}

#[test]
fn observer_propagates_into_nested_pipelines_with_dotted_paths() {
    block_on(async {
        let recorder = Arc::new(Recorder::default());
        let inner = PipelineBuilder::new().stage(PassThrough).build();
        let (ends, driver) = PipelineBuilder::new()
            .stage(inner)
            .observer(recorder.clone())
            .build()
            .start();
        let input = ends.input;
        let _output = ends.output;

        let feeder = async move {
            input
                .send_data(Transcript::user_final("deep").into())
                .await
                .unwrap();
        };
        join(feeder, driver).await;

        let seen = recorder.take();
        // Two wired reports: outer (nested pipeline + tail), then inner ("0.0").
        assert!(
            seen.iter().any(|s| matches!(
                s,
                Seen::Wired(ids) if ids.iter().any(|(p, _)| p == "0.0")
            )),
            "inner pipeline must report a dotted-path wiring, got {seen:?}",
        );
        // The frame is observed at the nested leaf and at the outer tail.
        assert!(
            seen.contains(&Seen::DataIn("0.0".into(), "pass")),
            "frame must be observed inside the nested pipeline, got {seen:?}",
        );
        assert!(
            seen.contains(&Seen::DataIn("1".into(), TAIL_STAGE)),
            "frame must be observed at the outer tail, got {seen:?}",
        );
    });
}

/// `perform` parks forever; an Interrupt must close the observer's open pair
/// with `PerformOutcome::Aborted`.
struct Parked {
    block_rx: Mutex<Option<oneshot::Receiver<()>>>,
    started: mpsc::Sender<()>,
}

impl Processor for Parked {
    type Effect = ();
    fn name(&self) -> &'static str {
        "parked"
    }
    fn decide_data(&mut self, _frame: &DataFrame) -> Decision<()> {
        Decision::drop().emit(())
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Stage for Parked {
    async fn perform(&self, _effect: (), _out: &Outbound) -> Result<(), StageError> {
        let _ = self.started.clone().send(()).await;
        let rx = self
            .block_rx
            .lock()
            .unwrap()
            .take()
            .expect("perform runs once");
        let _ = rx.await;
        Ok(())
    }
}

#[test]
fn interrupt_reports_aborted_perform() {
    block_on(async {
        let recorder = Arc::new(Recorder::default());
        let (started_tx, mut started_rx) = mpsc::channel::<()>(1);
        let (_block_tx, block_rx) = oneshot::channel::<()>();
        let stage = Parked {
            block_rx: Mutex::new(Some(block_rx)),
            started: started_tx,
        };
        let (ends, driver) = PipelineBuilder::new()
            .stage(stage)
            .observer(recorder.clone())
            .build()
            .start();
        let input = ends.input;
        let _output = ends.output;

        let feeder = async move {
            input
                .send_data(Transcript::user_final("go").into())
                .await
                .unwrap();
            started_rx.next().await.expect("perform must start");
            input
                .send_system(Direction::Down, SystemFrame::Interrupt)
                .await
                .unwrap();
        };
        join(feeder, driver).await;

        let seen = recorder.take();
        let start = seen
            .iter()
            .position(|s| *s == Seen::PerformStart("0".into()))
            .expect("perform_start must be reported");
        assert_eq!(
            seen[start + 1],
            Seen::PerformEnd("0".into(), PerformOutcome::Aborted),
            "the dropped perform must be closed as Aborted, got {seen:?}",
        );
        // The Interrupt itself is still observed on the system lane afterward.
        assert!(
            seen[start + 1..].contains(&Seen::SysIn("0".into(), "Interrupt")),
            "the interrupt must be observed after the abort, got {seen:?}",
        );
    });
}
