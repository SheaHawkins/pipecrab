//! `DispatchEgress` is a pure translator: `dispatch_task` / `update_task` calls
//! become `DispatchCommand`s that reach the sink and are echoed downstream as
//! native dispatch frames; unknown tools pass through untouched; malformed
//! dispatch calls become a recoverable error and are not sent. It holds no state,
//! so successive calls do not interfere.

mod common;

use std::sync::Arc;

use common::RecordingSink;
use futures::executor::block_on;
use pipecrab_core::{
    DataFrame, Decision, DispatchCommand, DispatchFrame, Disposition, Processor, ToolCall,
    Transcript,
};
use pipecrab_dispatch::{DispatchEgress, DispatchError, Publish};
use pipecrab_runtime::{Received, Stage, StageError, link};
use serde_json::json;

// --- Helpers. ----------------------------------------------------------------

fn tool_frame(id: &str, name: &str, args: serde_json::Value) -> DataFrame {
    DataFrame::from(ToolCall {
        id: Arc::from(id),
        name: Arc::from(name),
        arguments_json: Arc::from(args.to_string()),
    })
}

/// A tool call with verbatim (possibly invalid) argument text.
fn raw_tool_frame(id: &str, name: &str, args_json: &str) -> DataFrame {
    DataFrame::from(ToolCall {
        id: Arc::from(id),
        name: Arc::from(name),
        arguments_json: Arc::from(args_json),
    })
}

/// The single effect a decision emitted.
fn only(decision: Decision<Publish>) -> Publish {
    assert_eq!(
        decision.disposition,
        Disposition::Forward,
        "the tool call always forwards, staying visible downstream"
    );
    let mut effects = decision.effects;
    assert_eq!(effects.len(), 1, "expected exactly one effect");
    effects.pop().unwrap()
}

/// Perform one effect and return `(result, frames emitted downstream)`.
async fn run_effect(
    egress: &DispatchEgress<RecordingSink>,
    effect: Publish,
) -> (Result<(), StageError>, Vec<DataFrame>) {
    let (out, mut inbound) = link(16);
    let result = egress.perform(effect, &out).await;
    drop(out);
    let mut frames = Vec::new();
    while let Some(received) = inbound.recv().await {
        if let Received::Data(frame) = received {
            frames.push(frame);
        }
    }
    (result, frames)
}

// --- Translation. ------------------------------------------------------------

#[test]
fn dispatch_task_arguments_parse_into_a_create() {
    let (sink, _log) = RecordingSink::new();
    let mut egress = DispatchEgress::new(sink);

    let effect = only(egress.decide_data(&tool_frame(
        "call-1",
        "dispatch_task",
        json!({ "task": "book a flight", "context": "window seat" }),
    )));

    match effect {
        Publish::Command(DispatchCommand::Create {
            tool_call_id,
            task,
            context,
        }) => {
            assert_eq!(&*tool_call_id, "call-1");
            assert_eq!(&*task, "book a flight");
            assert_eq!(context.as_deref(), Some("window seat"));
        }
        other => panic!("expected a Create command, got {other:?}"),
    }
}

#[test]
fn dispatch_task_context_is_optional() {
    let (sink, _log) = RecordingSink::new();
    let mut egress = DispatchEgress::new(sink);

    let effect = only(egress.decide_data(&tool_frame(
        "call-1",
        "dispatch_task",
        json!({ "task": "check the weather" }),
    )));

    match effect {
        Publish::Command(DispatchCommand::Create { context, .. }) => assert_eq!(context, None),
        other => panic!("expected a Create command, got {other:?}"),
    }
}

#[test]
fn update_task_arguments_parse_into_an_update() {
    let (sink, _log) = RecordingSink::new();
    let mut egress = DispatchEgress::new(sink);

    let effect = only(egress.decide_data(&tool_frame(
        "call-9",
        "update_task",
        json!({ "task_id": "task-42", "message": "any update?" }),
    )));

    match effect {
        Publish::Command(DispatchCommand::Update {
            tool_call_id,
            task_id,
            message,
        }) => {
            assert_eq!(&*tool_call_id, "call-9");
            assert_eq!(&*task_id, "task-42");
            assert_eq!(&*message, "any update?");
        }
        other => panic!("expected an Update command, got {other:?}"),
    }
}

#[test]
fn unknown_tools_are_forwarded_untouched() {
    let (sink, _log) = RecordingSink::new();
    let mut egress = DispatchEgress::new(sink);

    let decision = egress.decide_data(&tool_frame("call-1", "some_other_tool", json!({ "x": 1 })));
    assert_eq!(
        decision.disposition,
        Disposition::Forward,
        "an unknown tool passes through"
    );
    assert!(
        decision.effects.is_empty(),
        "an unknown tool produces no dispatch effect"
    );
}

#[test]
fn non_tool_frames_forward_untouched() {
    let (sink, _log) = RecordingSink::new();
    let mut egress = DispatchEgress::new(sink);

    let decision = egress.decide_data(&Transcript::agent_final("hello").into());
    assert_eq!(decision.disposition, Disposition::Forward);
    assert!(decision.effects.is_empty());
}

#[test]
fn malformed_dispatch_arguments_reject_and_send_nothing() {
    block_on(async {
        let (sink, log) = RecordingSink::new();
        let mut egress = DispatchEgress::new(sink);

        // Missing the required `task` field.
        let effect = only(egress.decide_data(&tool_frame(
            "call-1",
            "dispatch_task",
            json!({ "context": "no task here" }),
        )));
        assert!(matches!(effect, Publish::Reject(_)), "malformed → reject");

        // Performing a reject is a recoverable error and sends nothing.
        let (result, frames) = run_effect(&egress, effect).await;
        let err = result.unwrap_err();
        assert!(!err.fatal, "a malformed call is recoverable");
        assert!(frames.is_empty(), "no command frame is emitted");
        assert!(log.lock().unwrap().is_empty(), "the sink is never called");

        // Invalid JSON text rejects the same way.
        let effect =
            only(egress.decide_data(&raw_tool_frame("call-2", "update_task", "{ not json")));
        assert!(matches!(effect, Publish::Reject(_)));
    });
}

// --- Publication and transport. ----------------------------------------------

#[test]
fn a_valid_command_reaches_the_sink_and_is_echoed_downstream() {
    block_on(async {
        let (sink, log) = RecordingSink::new();
        let mut egress = DispatchEgress::new(sink);

        let call = tool_frame("call-1", "dispatch_task", json!({ "task": "ping" }));
        // The tool call itself forwards downstream (stays visible to observers).
        let decision = egress.decide_data(&call);
        assert_eq!(decision.disposition, Disposition::Forward);
        let effect = only(decision);

        let (result, frames) = run_effect(&egress, effect).await;
        result.unwrap();

        // The command reached the sink...
        let sent = log.lock().unwrap().clone();
        assert_eq!(sent.len(), 1);
        assert!(matches!(&sent[0], DispatchCommand::Create { task, .. } if &**task == "ping"));

        // ...and was echoed downstream as a native dispatch frame.
        assert_eq!(frames.len(), 1);
        assert!(matches!(
            &frames[0],
            DataFrame::Dispatch(DispatchFrame::Command(DispatchCommand::Create { task, .. }))
                if &**task == "ping"
        ));
    });
}

#[test]
fn a_sink_error_becomes_a_stage_error_with_the_transport_classification() {
    block_on(async {
        // A fatal transport error tears the pipeline down.
        let (sink, log) = RecordingSink::failing(DispatchError::fatal("socket closed"));
        let mut egress = DispatchEgress::new(sink);
        let effect =
            only(egress.decide_data(&tool_frame("c", "dispatch_task", json!({ "task": "x" }))));
        let (result, frames) = run_effect(&egress, effect).await;
        let err = result.unwrap_err();
        assert!(err.fatal, "a fatal transport error stays fatal");
        assert!(
            frames.is_empty(),
            "a failed send emits no downstream command"
        );
        assert!(log.lock().unwrap().is_empty());

        // A recoverable transport error keeps the pipeline alive.
        let (sink, _log) = RecordingSink::failing(DispatchError::recoverable("dropped"));
        let mut egress = DispatchEgress::new(sink);
        let effect =
            only(egress.decide_data(&tool_frame("c", "dispatch_task", json!({ "task": "x" }))));
        let (result, _frames) = run_effect(&egress, effect).await;
        assert!(
            !result.unwrap_err().fatal,
            "a recoverable error stays recoverable"
        );
    });
}

#[test]
fn egress_retains_no_state_across_calls() {
    // Egress holds no task map: successive dispatch calls translate
    // independently, each carrying its own tool_call_id, with no interference.
    let (sink, _log) = RecordingSink::new();
    let mut egress = DispatchEgress::new(sink);

    let first = only(egress.decide_data(&tool_frame(
        "call-1",
        "dispatch_task",
        json!({ "task": "a" }),
    )));
    let second = only(egress.decide_data(&tool_frame(
        "call-2",
        "dispatch_task",
        json!({ "task": "b" }),
    )));

    let ids: Vec<String> = [first, second]
        .into_iter()
        .map(|e| match e {
            Publish::Command(DispatchCommand::Create { tool_call_id, .. }) => {
                tool_call_id.to_string()
            }
            other => panic!("expected a Create, got {other:?}"),
        })
        .collect();
    assert_eq!(ids, ["call-1", "call-2"]);
}
