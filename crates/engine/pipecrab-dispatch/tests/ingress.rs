//! `DispatchIngress` is the active head stage: its single pipeline driver pumps
//! the external source alongside the lanes. Every external event enters as a raw
//! `Dispatch` frame followed by its model projection; ordinary frames pass
//! through; interrupts don't stop listening; `Stop` cancels the source.
//!
//! Deterministic and tokio-free (`block_on`): the source is driven over a
//! channel, and each test terminates ingress on a handshake (drop the input once
//! the expected frames have drained) or an explicit `Stop`.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use common::{ScriptedSource, SourceStep};
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::sink::SinkExt;
use pipecrab_core::{
    DataFrame, Direction, DispatchEvent, DispatchFrame, ModelFrame, ModelInput, ModelMessage,
    SystemFrame, Transcript,
};
use pipecrab_dispatch::DispatchIngress;
use pipecrab_runtime::{Inbound, PipelineBuilder, Received};

// --- Harness. ----------------------------------------------------------------

/// Collect output frames until the pipeline closes. After `expect` *data* frames
/// have arrived, signal `done` so the feeder can terminate ingress.
async fn drain(
    mut output: Inbound,
    expect: usize,
    done: oneshot::Sender<()>,
) -> (Vec<DataFrame>, Vec<SystemFrame>) {
    let mut data = Vec::new();
    let mut sys = Vec::new();
    let mut done = Some(done);
    while let Some(received) = output.recv().await {
        match received {
            Received::Data(frame) => {
                data.push(frame);
                if data.len() == expect {
                    if let Some(d) = done.take() {
                        let _ = d.send(());
                    }
                }
            }
            Received::Sys(_dir, frame) => sys.push(frame),
        }
    }
    (data, sys)
}

// --- Event constructors. -----------------------------------------------------

fn accepted(call: &str, task: &str) -> DispatchEvent {
    DispatchEvent::Accepted {
        tool_call_id: Arc::from(call),
        task_id: Arc::from(task),
    }
}
fn rejected(call: &str, message: &str) -> DispatchEvent {
    DispatchEvent::Rejected {
        tool_call_id: Arc::from(call),
        message: Arc::from(message),
    }
}
fn progress(task: &str, message: &str) -> DispatchEvent {
    DispatchEvent::Progress {
        task_id: Arc::from(task),
        message: Arc::from(message),
    }
}
fn question(task: &str, message: &str) -> DispatchEvent {
    DispatchEvent::Question {
        task_id: Arc::from(task),
        message: Arc::from(message),
    }
}
fn completion(task: &str, message: &str) -> DispatchEvent {
    DispatchEvent::Completion {
        task_id: Arc::from(task),
        message: Arc::from(message),
    }
}
fn failure(task: &str, message: &str) -> DispatchEvent {
    DispatchEvent::Failure {
        task_id: Arc::from(task),
        message: Arc::from(message),
        retryable: false,
    }
}

// --- Frame extractors. -------------------------------------------------------

fn raw_event(frame: &DataFrame) -> &DispatchEvent {
    match frame {
        DataFrame::Dispatch(DispatchFrame::Event(event)) => event,
        other => panic!("expected a raw Dispatch event, got {other:?}"),
    }
}

fn projection(frame: &DataFrame) -> &ModelInput {
    match frame {
        DataFrame::Model(ModelFrame::Input(input)) => input,
        other => panic!("expected a model projection, got {other:?}"),
    }
}

// --- Tests. ------------------------------------------------------------------

#[test]
fn the_single_driver_pumps_the_source_while_the_data_lane_is_idle() {
    // No ordinary frames flow; the pipeline's one driver still receives external
    // events — there is no separate listener future.
    block_on(async {
        let (source, mut tx, _cancels) = ScriptedSource::new();
        let (ends, driver) = PipelineBuilder::new()
            .stage(DispatchIngress::new(source))
            .build()
            .start();
        let input = ends.input;
        let (done_tx, done_rx) = oneshot::channel();

        let feed = async move {
            tx.send(SourceStep::Event(accepted("call-1", "task-1")))
                .await
                .unwrap();
            tx.send(SourceStep::Event(progress("task-1", "halfway")))
                .await
                .unwrap();
            done_rx.await.ok();
            drop(input); // both lanes close → ingress terminates
        };

        let (_, (data, _sys), _) = futures::join!(feed, drain(ends.output, 4, done_tx), driver);

        // Two events → four data frames: each raw event then its projection.
        assert_eq!(data.len(), 4);
        assert!(matches!(
            raw_event(&data[0]),
            DispatchEvent::Accepted { .. }
        ));
        assert!(matches!(
            raw_event(&data[2]),
            DispatchEvent::Progress { .. }
        ));
    });
}

#[test]
fn the_raw_event_precedes_its_projection() {
    block_on(async {
        let (source, mut tx, _cancels) = ScriptedSource::new();
        let (ends, driver) = PipelineBuilder::new()
            .stage(DispatchIngress::new(source))
            .build()
            .start();
        let input = ends.input;
        let (done_tx, done_rx) = oneshot::channel();

        let feed = async move {
            tx.send(SourceStep::Event(accepted("call-1", "task-7")))
                .await
                .unwrap();
            done_rx.await.ok();
            drop(input);
        };

        let (_, (data, _sys), _) = futures::join!(feed, drain(ends.output, 2, done_tx), driver);

        assert_eq!(data.len(), 2);
        // frame 0 is the authoritative raw event; frame 1 is the projection.
        assert!(matches!(
            raw_event(&data[0]),
            DispatchEvent::Accepted { .. }
        ));
        assert!(matches!(projection(&data[1]), ModelInput::Context(_)));
    });
}

#[test]
fn each_event_projects_to_its_documented_model_input() {
    block_on(async {
        let (source, mut tx, _cancels) = ScriptedSource::new();
        let (ends, driver) = PipelineBuilder::new()
            .stage(DispatchIngress::new(source))
            .build()
            .start();
        let input = ends.input;
        let (done_tx, done_rx) = oneshot::channel();

        let events = [
            accepted("call-1", "task-1"),
            rejected("call-2", "quota exceeded"),
            progress("task-1", "50%"),
            question("task-1", "which seat?"),
            completion("task-1", "booked"),
            failure("task-1", "timeout"),
        ];
        let feed = async move {
            for event in events {
                tx.send(SourceStep::Event(event)).await.unwrap();
            }
            done_rx.await.ok();
            drop(input);
        };

        // 6 events → 12 frames.
        let (_, (data, _sys), _) = futures::join!(feed, drain(ends.output, 12, done_tx), driver);
        assert_eq!(data.len(), 12);

        // Projections are at the odd indices; raw events at the even ones.
        // Accepted → context tool result carrying the assigned task_id.
        match projection(&data[1]) {
            ModelInput::Context(ModelMessage::ToolResult {
                tool_call_id,
                name,
                content,
            }) => {
                assert_eq!(&**tool_call_id, "call-1");
                assert_eq!(&**name, "dispatch_task");
                let parsed: serde_json::Value = serde_json::from_str(content).unwrap();
                assert_eq!(
                    parsed["task_id"], "task-1",
                    "the result must include the task_id"
                );
            }
            other => panic!("Accepted must project to a context tool result, got {other:?}"),
        }
        // Rejected → responding event.
        assert_event(projection(&data[3]), Respond, "rejected");
        // Progress → context-only event (visible next turn, no interruption).
        assert_event(projection(&data[5]), Context, "progress");
        // Question / Completion / Failure → responding events.
        assert_event(projection(&data[7]), Respond, "question");
        assert_event(projection(&data[9]), Respond, "completion");
        assert_event(projection(&data[11]), Respond, "failure");
    });
}

#[derive(PartialEq)]
enum Kind {
    Context,
    Respond,
}
use Kind::{Context, Respond};

/// Assert a projection is a dispatch `Event` of the given kind and mode.
fn assert_event(input: &ModelInput, mode: Kind, kind: &str) {
    let message = match input {
        ModelInput::Context(m) => {
            assert!(mode == Context, "expected a responding event, got context");
            m
        }
        ModelInput::Respond(m) => {
            assert!(mode == Respond, "expected a context event, got responding");
            m
        }
    };
    match message {
        ModelMessage::Event {
            source,
            kind: event_kind,
            ..
        } => {
            assert_eq!(&**source, "dispatch");
            assert_eq!(&**event_kind, kind);
        }
        other => panic!("expected a dispatch Event, got {other:?}"),
    }
}

#[test]
fn ordinary_inbound_frames_pass_through() {
    block_on(async {
        let (source, _tx, _cancels) = ScriptedSource::new();
        let (ends, driver) = PipelineBuilder::new()
            .stage(DispatchIngress::new(source))
            .build()
            .start();
        let input = ends.input;
        let (done_tx, done_rx) = oneshot::channel();

        let feed = async move {
            let _ = input
                .send_data(Transcript::user_final("hello").into())
                .await;
            done_rx.await.ok();
            drop(input);
        };

        let (_, (data, _sys), _) = futures::join!(feed, drain(ends.output, 1, done_tx), driver);

        assert_eq!(data.len(), 1);
        assert!(matches!(
            &data[0],
            DataFrame::Transcript(t) if &*t.text == "hello"
        ));
    });
}

#[test]
fn dispatch_listening_continues_after_an_interrupt() {
    block_on(async {
        let (source, mut tx, _cancels) = ScriptedSource::new();
        let (ends, driver) = PipelineBuilder::new()
            .stage(DispatchIngress::new(source))
            .build()
            .start();
        let input = ends.input;
        let (done_tx, done_rx) = oneshot::channel();

        let feed = async move {
            let _ = input
                .send_system(Direction::Down, SystemFrame::Interrupt)
                .await;
            // An event *after* the interrupt still flows: listening continues.
            tx.send(SourceStep::Event(completion("task-1", "done")))
                .await
                .unwrap();
            done_rx.await.ok();
            drop(input);
        };

        let (_, (data, sys), _) = futures::join!(feed, drain(ends.output, 2, done_tx), driver);

        // The post-interrupt event was received and projected.
        assert_eq!(data.len(), 2);
        assert!(matches!(
            raw_event(&data[0]),
            DispatchEvent::Completion { .. }
        ));
        // The interrupt was forwarded downstream.
        assert!(sys.iter().any(|f| matches!(f, SystemFrame::Interrupt)));
    });
}

#[test]
fn stop_cancels_the_source_and_terminates() {
    block_on(async {
        // Keep `tx` alive so the source stays *open* (parked): only the Stop, not
        // source closure, ends ingress.
        let (source, _tx, cancels) = ScriptedSource::new();
        let (ends, driver) = PipelineBuilder::new()
            .stage(DispatchIngress::new(source))
            .build()
            .start();
        let input = ends.input;
        let (done_tx, _done_rx) = oneshot::channel();

        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Stop).await;
            drop(input);
        };

        let (_, (_data, sys), _) =
            futures::join!(feed, drain(ends.output, usize::MAX, done_tx), driver);

        assert_eq!(
            cancels.load(Ordering::SeqCst),
            1,
            "Stop cancels the source once"
        );
        assert!(
            sys.iter().any(|f| matches!(f, SystemFrame::Stop)),
            "Stop is forwarded"
        );
    });
}

#[test]
fn source_closure_leaves_the_pipeline_input_open() {
    block_on(async {
        let (source, tx, cancels) = ScriptedSource::new();
        let (ends, driver) = PipelineBuilder::new()
            .stage(DispatchIngress::new(source))
            .build()
            .start();
        let input = ends.input;
        let (done_tx, done_rx) = oneshot::channel();

        let feed = async move {
            drop(tx); // close the source
            // The pipeline input still carries frames after the source closed.
            let _ = input
                .send_data(Transcript::user_final("still here").into())
                .await;
            done_rx.await.ok();
            drop(input);
        };

        let (_, (data, _sys), _) = futures::join!(feed, drain(ends.output, 1, done_tx), driver);

        assert_eq!(data.len(), 1);
        assert!(matches!(&data[0], DataFrame::Transcript(t) if &*t.text == "still here"));
        assert_eq!(
            cancels.load(Ordering::SeqCst),
            1,
            "cancelled once, on final teardown"
        );
    });
}

#[test]
fn a_recoverable_source_error_surfaces_and_listening_continues() {
    block_on(async {
        let (source, mut tx, _cancels) = ScriptedSource::new();
        let (ends, driver) = PipelineBuilder::new()
            .stage(DispatchIngress::new(source))
            .build()
            .start();
        let input = ends.input;
        let (done_tx, done_rx) = oneshot::channel();

        let feed = async move {
            tx.send(SourceStep::Fail(
                pipecrab_dispatch::DispatchError::recoverable("dropped frame"),
            ))
            .await
            .unwrap();
            // After a recoverable error, a later event still flows.
            tx.send(SourceStep::Event(progress("task-1", "resumed")))
                .await
                .unwrap();
            done_rx.await.ok();
            drop(input);
        };

        let (_, (data, sys), _) = futures::join!(feed, drain(ends.output, 2, done_tx), driver);

        // The event after the error arrived.
        assert_eq!(data.len(), 2);
        assert!(matches!(
            raw_event(&data[0]),
            DispatchEvent::Progress { .. }
        ));
        // The error surfaced as a non-fatal Error frame travelling upstream.
        assert!(sys.iter().any(|f| matches!(
            f,
            SystemFrame::Error { fatal, .. } if !fatal
        )));
    });
}

#[test]
fn a_fatal_source_error_terminates_ingress() {
    block_on(async {
        let (source, mut tx, cancels) = ScriptedSource::new();
        let (ends, driver) = PipelineBuilder::new()
            .stage(DispatchIngress::new(source))
            .build()
            .start();
        // Held open for the whole test: only the fatal error should end ingress,
        // not a closed pipeline input.
        let _input = ends.input;
        let (done_tx, _done_rx) = oneshot::channel();

        let feed = async move {
            tx.send(SourceStep::Fail(pipecrab_dispatch::DispatchError::fatal(
                "socket closed",
            )))
            .await
            .unwrap();
        };

        let (_, (_data, sys), _) =
            futures::join!(feed, drain(ends.output, usize::MAX, done_tx), driver);

        assert!(
            sys.iter()
                .any(|f| matches!(f, SystemFrame::Error { fatal, .. } if *fatal)),
            "the fatal error surfaces"
        );
        assert_eq!(
            cancels.load(Ordering::SeqCst),
            1,
            "a fatal error cancels the source"
        );
    });
}
