//! One genuine round trip through the framework: a `dispatch_task` tool call fed
//! into a `PipelineBuilder`-built pipeline of the real
//! [`DispatchIngress`]/[`DispatchEgress`] stages — obtained from
//! `Dispatch::new(source, sink).into_stages()` — reaches this transport's sink,
//! posts a run, and the poller's completion comes back out past the tail.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{Sequence, completed, config_for, run_status, started};
use pipecrab_core::{DataFrame, DispatchEvent, DispatchFrame, ToolCall};
use pipecrab_dispatch::Dispatch;
use pipecrab_dispatch_hermes::{TaskId, connect};
use pipecrab_runtime::{PipelineBuilder, Received};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn a_tool_call_drives_the_transport_end_to_end() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/runs"))
        .respond_with(ResponseTemplate::new(202).set_body_json(started("run_e2e")))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/runs/run_e2e"))
        .respond_with(Sequence::new(
            200,
            vec![
                run_status("run_e2e", "running"),
                completed("run_e2e", "pipeline done"),
            ],
        ))
        .mount(&server)
        .await;

    let (source, sink) = connect(config_for(&server));
    let dispatch = Dispatch::new(source, sink.clone());

    // The tools wire in exactly as documented.
    let names: Vec<Arc<str>> = dispatch
        .tool_definitions()
        .iter()
        .map(|t| t.name.clone())
        .collect();
    assert_eq!(
        names.iter().map(|n| &**n).collect::<Vec<_>>(),
        ["dispatch_task", "update_task"]
    );

    let (ingress, egress) = dispatch.into_stages();
    let (ends, driver) = PipelineBuilder::new()
        .stage(ingress)
        .stage(egress)
        .build()
        .start();
    let driver = tokio::spawn(driver);

    // Feed a dispatch_task tool call at the head of the pipeline.
    ends.input
        .send_data(DataFrame::from(ToolCall {
            id: Arc::from("call-e2e"),
            name: Arc::from("dispatch_task"),
            arguments_json: Arc::from(json!({ "task": "run the pipeline" }).to_string()),
        }))
        .await
        .expect("feed the tool call");

    // Drain past the tail until the run's completion emerges, noting the task id
    // the acceptance assigned along the way.
    let mut output = ends.output;
    let mut accepted_task: Option<TaskId> = None;
    let mut completion_message: Option<String> = None;
    while completion_message.is_none() {
        let received = tokio::time::timeout(Duration::from_secs(5), output.recv())
            .await
            .expect("a frame within budget");
        match received {
            Some(Received::Data(DataFrame::Dispatch(DispatchFrame::Event(event)))) => match event {
                DispatchEvent::Accepted { task_id, .. } => {
                    accepted_task = Some(TaskId::from(task_id));
                }
                DispatchEvent::Completion { message, .. } => {
                    completion_message = Some(message.to_string());
                }
                _ => {}
            },
            Some(_) => {}
            None => panic!("pipeline output closed before the completion arrived"),
        }
    }

    assert_eq!(completion_message.as_deref(), Some("pipeline done"));
    let task_id = accepted_task.expect("acceptance projected a task_id");

    // The sink handle can still read that task's trail.
    let ring = sink.task_events(&task_id).expect("known task");
    assert!(
        ring.iter()
            .any(|(_, e)| matches!(e, DispatchEvent::Completion { .. })),
        "the completion is in the task's ring"
    );

    // Closing the input lane tears the pipeline down; the driver then finishes.
    drop(output);
    drop(ends.input);
    let _ = tokio::time::timeout(Duration::from_secs(5), driver).await;
}
