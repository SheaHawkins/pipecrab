//! The tool definitions and the `Dispatch` composition helper: the two tools are
//! valid, provider-neutral, and carry the model guidance the descriptions must;
//! `Dispatch::new` hands out the tools and splits into the two stages.

mod common;

use common::{RecordingSink, ScriptedSource};
use pipecrab_dispatch::{
    Dispatch, dispatch_task_definition, dispatch_tool_definitions, update_task_definition,
};

#[test]
fn dispatch_task_definition_has_the_documented_shape() {
    let def = dispatch_task_definition();
    assert_eq!(&*def.name, "dispatch_task");

    let params = &def.parameters;
    // task is a required string; context is an optional (nullable) string.
    assert_eq!(params["properties"]["task"]["type"], "string");
    assert_eq!(params["required"], serde_json::json!(["task"]));
    assert_eq!(
        params["properties"]["context"]["type"],
        serde_json::json!(["string", "null"])
    );
}

#[test]
fn update_task_definition_has_the_documented_shape() {
    let def = update_task_definition();
    assert_eq!(&*def.name, "update_task");

    let params = &def.parameters;
    assert_eq!(params["properties"]["task_id"]["type"], "string");
    assert_eq!(params["properties"]["message"]["type"], "string");
    assert_eq!(
        params["required"],
        serde_json::json!(["task_id", "message"])
    );
}

#[test]
fn descriptions_instruct_the_model() {
    let dispatch = dispatch_task_definition().description.to_lowercase();
    let update = update_task_definition().description.to_lowercase();

    // dispatch_task begins a new task; update_task communicates with an existing one.
    assert!(dispatch.contains("new") && dispatch.contains("task"));
    assert!(update.contains("existing"));

    // Both: acknowledge out loud first, and don't claim success before completion.
    for d in [&dispatch, &update] {
        assert!(
            d.contains("acknowledg"),
            "a description must ask for a spoken acknowledgement first: {d:?}"
        );
        assert!(
            d.contains("completion"),
            "a description must forbid claiming success before a completion event: {d:?}"
        );
    }
}

#[test]
fn dispatch_tool_definitions_bundles_both_tools() {
    let tools = dispatch_tool_definitions();
    let names: Vec<&str> = tools.iter().map(|t| &*t.name).collect();
    assert_eq!(names, ["dispatch_task", "update_task"]);
}

#[test]
fn compose_exposes_tools_and_splits_into_stages() {
    let (source, _tx, _cancels) = ScriptedSource::new();
    let (sink, _log) = RecordingSink::new();

    let dispatch = Dispatch::new(source, sink);
    let tools = dispatch.tool_definitions();
    let names: Vec<&str> = tools.iter().map(|t| &*t.name).collect();
    assert_eq!(names, ["dispatch_task", "update_task"]);

    // Splits into exactly the two stages; both move their transport half in.
    let (_ingress, _egress) = dispatch.into_stages();
}
