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
        // Leads with the capability, because the failure this description has to
        // beat is the model refusing ("I can't check the weather") instead of
        // reaching for the tool that can.
        "Hand a question or errand to a background agent that can browse the \
         web and use tools — it can find out today's weather, the news, \
         prices, schedules, and anything else you do not already know. Call \
         this instead of telling the user you do not know, cannot check, or \
         have no access. Never ask the user to restate or narrow the errand: \
         the request you already have is the task. It runs asynchronously: do \
         NOT claim it has succeeded, finished, or produced a result until a \
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
         `task_id` it was assigned when it was accepted. Call this only for a \
         task already accepted in this conversation, whose `task_id` you \
         therefore already have — never ask the user for a `task_id`, and \
         never use this to start work: starting is `dispatch_task`. Do NOT \
         claim the task has succeeded or finished until a completion event \
         arrives.",
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
