//! The two dispatch tool definitions, as ordinary [`pipecrab_lm::ToolDefinition`]
//! values: [`dispatch_task_definition`] starts a task, [`update_task_definition`]
//! messages an existing one. [`dispatch_tool_definitions`] bundles both for
//! [`LmStage::with_tools`](pipecrab_lm::LmStage::with_tools).
//!
//! No Rig or other provider type appears here — these are the provider-neutral
//! shapes every hosted adapter turns into its own tool format.

use std::sync::Arc;

use pipecrab_lm::ToolDefinition;
use serde_json::json;

/// The `dispatch_task` tool: begin a new asynchronous background task.
///
/// Arguments: `task` (required string) and `context` (optional string).
pub fn dispatch_task_definition() -> ToolDefinition {
    ToolDefinition::new(
        "dispatch_task",
        // The description is the model's contract: acknowledge out loud first,
        // and never announce success before a completion event arrives.
        "Begin a new asynchronous background task. Speak a brief spoken \
         acknowledgement to the user *before* calling this tool — the call \
         itself produces no speech. The task then runs asynchronously: do NOT \
         claim it has succeeded, finished, or produced a result until a \
         completion event arrives. Put the work to perform in `task`, and any \
         extra detail in `context`.",
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The work the new task should perform."
                },
                "context": {
                    "type": ["string", "null"],
                    "description": "Optional additional context for the task."
                }
            },
            "required": ["task"]
        }),
    )
    .expect("the static dispatch_task definition is valid")
}

/// The `update_task` tool: send a follow-up to an existing asynchronous task.
///
/// Arguments: `task_id` (required string) and `message` (required string).
pub fn update_task_definition() -> ToolDefinition {
    ToolDefinition::new(
        "update_task",
        "Communicate with an existing asynchronous task, identified by the \
         `task_id` it was assigned when it was accepted. Speak a brief spoken \
         acknowledgement to the user *before* calling this tool — the call \
         itself produces no speech. Do NOT claim the task has succeeded or \
         finished until a completion event arrives.",
        json!({
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Identifier of the task to message, from its acceptance."
                },
                "message": {
                    "type": "string",
                    "description": "The follow-up message for the task."
                }
            },
            "required": ["task_id", "message"]
        }),
    )
    .expect("the static update_task definition is valid")
}

/// Both dispatch tools, ready to hand to
/// [`LmStage::with_tools`](pipecrab_lm::LmStage::with_tools).
pub fn dispatch_tool_definitions() -> Arc<[ToolDefinition]> {
    Arc::from([dispatch_task_definition(), update_task_definition()])
}
