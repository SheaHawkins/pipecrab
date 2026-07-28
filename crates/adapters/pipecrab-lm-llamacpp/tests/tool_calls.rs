//! Tool calling against a real model: a turn that wants a tool produces one
//! `ModelDelta::ToolCall` and no spoken call syntax, an ordinary turn is
//! untouched, and a tool result round-trips back into the prompt.
//!
//! A separate integration-test file on purpose: llama.cpp's backend is a
//! process-global singleton, and each test file runs as its own process.

use std::num::NonZeroU32;
use std::sync::Arc;

use futures::{StreamExt, executor::block_on};
use pipecrab_lm::{
    Conversation, GenParams, LanguageModel, Message, ModelDelta, ToolCall, ToolDefinition,
};
use pipecrab_lm_llamacpp::{LlamaCpp, LlamaCppConfig};
use serde_json::json;

/// The dispatch tool surface, restated here so the adapter's tests do not depend
/// on `pipecrab-dispatch`.
fn tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "dispatch_task",
            "Begin a new asynchronous background task. Put the work to perform in \
             `task`, and any extra detail in `context`.",
            json!({
                "type": "object",
                "properties": {
                    "task": { "type": "string", "description": "The work the new task should perform." },
                    "context": { "type": ["string", "null"], "description": "Optional extra context." },
                },
                "required": ["task"],
            }),
        )
        .expect("valid tool"),
        ToolDefinition::new(
            "update_task",
            "Send a follow-up to an existing asynchronous task, identified by its `task_id`.",
            json!({
                "type": "object",
                "properties": {
                    "task_id": { "type": "string", "description": "Identifier of the task." },
                    "message": { "type": "string", "description": "The follow-up message." },
                },
                "required": ["task_id", "message"],
            }),
        )
        .expect("valid tool"),
    ]
}

fn model() -> LlamaCpp {
    let path = std::env::var_os("PIPECRAB_LLAMA_MODEL")
        .expect("PIPECRAB_LLAMA_MODEL must point to a chat GGUF");
    LlamaCpp::load(LlamaCppConfig::new(path).with_generation_defaults(
        NonZeroU32::new(192).expect("non-zero"),
        0.0,
        42,
    ))
    .expect("load model")
}

fn turn(
    model: &LlamaCpp,
    conversation: &Conversation,
    tools: &[ToolDefinition],
) -> (String, Vec<ToolCall>) {
    let deltas = block_on(async {
        model
            .generate(conversation, &GenParams::default(), tools)
            .await
            .expect("start generation")
            .collect::<Vec<_>>()
            .await
    });
    let mut text = String::new();
    let mut calls = Vec::new();
    for delta in deltas {
        match delta.expect("generation delta") {
            ModelDelta::Text(piece) => text.push_str(&piece),
            ModelDelta::ToolCall(call) => calls.push(call),
        }
    }
    (text, calls)
}

fn ask(user: &str) -> Conversation {
    Conversation {
        messages: vec![
            Message::system(
                "You are a helpful voice assistant. Replies are spoken aloud, so answer in \
                 one or two short sentences of plain prose.",
            ),
            Message::user(user),
        ],
    }
}

#[test]
#[ignore = "requires PIPECRAB_LLAMA_MODEL to point to a chat GGUF"]
fn a_tool_turn_emits_one_call_and_speaks_none_of_its_syntax() {
    let model = model();
    let (text, calls) = turn(
        &model,
        &ask("Please kick off a background task to research sea otters."),
        &tools(),
    );

    assert_eq!(calls.len(), 1, "expected one tool call, got {calls:?}");
    assert_eq!(&*calls[0].name, "dispatch_task");
    assert!(!calls[0].id.is_empty(), "a minted id is never empty");
    let arguments: serde_json::Value =
        serde_json::from_str(&calls[0].arguments_json).expect("arguments are JSON");
    assert!(
        arguments
            .get("task")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|task| !task.is_empty()),
        "the required `task` argument is missing: {arguments}"
    );

    // The whole point of intercepting mid-stream: anything left in the text is
    // spoken aloud by the TTS stage.
    for fragment in ["<tool_call>", "</tool_call>", "dispatch_task", "\"name\""] {
        assert!(
            !text.contains(fragment),
            "call syntax {fragment:?} reached the transcript: {text:?}"
        );
    }
}

#[test]
#[ignore = "requires PIPECRAB_LLAMA_MODEL to point to a chat GGUF"]
fn an_ordinary_turn_is_unconstrained_and_calls_nothing() {
    let model = model();
    let question = ask("Say hello in one short sentence.");

    let (reply, calls) = turn(&model, &question, &tools());
    assert!(calls.is_empty(), "an ordinary turn called {calls:?}");
    assert!(!reply.trim().is_empty(), "no reply was generated");
    assert!(
        !reply.contains("<tool") && !reply.contains("\"arguments\""),
        "an ordinary turn leaked call syntax: {reply:?}"
    );

    // The lazy grammar masks nothing until the trigger fires, so a turn that
    // never trips it samples exactly as it would without one. Declarations do
    // change the prompt, so the comparable run is the same prompt again.
    let (repeat, _) = turn(&model, &question, &tools());
    assert_eq!(reply, repeat, "the tool grammar perturbed a text-only turn");
}

#[test]
#[ignore = "requires PIPECRAB_LLAMA_MODEL to point to a chat GGUF"]
fn a_tool_result_round_trips_and_the_call_is_not_reissued() {
    let model = model();
    let tools = tools();
    let mut conversation = ask("Please kick off a background task to research sea otters.");

    let (text, calls) = turn(&model, &conversation, &tools);
    let call = calls.first().cloned().expect("a first-turn tool call");

    // Replay the assistant's call and its result, exactly as `LmStage` commits
    // them, then ask a question that needs no tool.
    conversation
        .messages
        .push(Message::assistant_with_tool_calls(text, [call.clone()]));
    conversation.messages.push(Message::ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        content: Arc::from("The task was accepted and is running."),
    });
    conversation
        .messages
        .push(Message::user("Thanks. Is it running?"));

    let (reply, calls) = turn(&model, &conversation, &tools);
    assert!(
        calls.is_empty(),
        "the model re-issued a call it had already made and had answered: {calls:?}"
    );
    assert!(!reply.trim().is_empty(), "no reply after the tool result");
}

#[test]
#[ignore = "requires PIPECRAB_LLAMA_MODEL to point to a chat GGUF"]
fn ids_are_unique_across_turns() {
    // The id becomes the dispatch idempotency key: a repeat between turns is
    // silently deduplicated by the gateway and that task never runs.
    let model = model();
    let tools = tools();
    let mut ids = Vec::new();
    for task in ["research sea otters", "summarise today's news"] {
        let (_, calls) = turn(
            &model,
            &ask(&format!("Please kick off a background task to {task}.")),
            &tools,
        );
        ids.extend(calls.into_iter().map(|call| call.id));
    }
    assert!(ids.len() >= 2, "expected a call per turn, got {ids:?}");
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(
        unique.len(),
        ids.len(),
        "ids collided across turns: {ids:?}"
    );
}

#[test]
#[ignore = "requires PIPECRAB_LLAMA_MODEL to point to a chat GGUF"]
fn a_call_truncated_by_max_tokens_fails_instead_of_half_dispatching() {
    let path = std::env::var_os("PIPECRAB_LLAMA_MODEL")
        .expect("PIPECRAB_LLAMA_MODEL must point to a chat GGUF");
    // Enough tokens to open the call, never enough to close it.
    let model = LlamaCpp::load(LlamaCppConfig::new(path).with_generation_defaults(
        NonZeroU32::new(8).expect("non-zero"),
        0.0,
        42,
    ))
    .expect("load model");

    let deltas = block_on(async {
        model
            .generate(
                &ask("Please kick off a background task to research sea otters."),
                &GenParams::default(),
                &tools(),
            )
            .await
            .expect("start generation")
            .collect::<Vec<_>>()
            .await
    });
    assert!(
        deltas.iter().any(Result::is_err),
        "a truncated call reported no error: {deltas:?}"
    );
    assert!(
        !deltas
            .iter()
            .any(|delta| matches!(delta, Ok(ModelDelta::ToolCall(_)))),
        "a partial call was emitted: {deltas:?}"
    );
}

/// Qwen 3's non-thinking mode, verbatim from its chat template's
/// `enable_thinking=false` branch.
const NO_THINK: &str = "<think>\n\n</think>\n\n";

/// A reasoning GGUF — Qwen 3 or another family that thinks before answering.
/// Separate from `PIPECRAB_LLAMA_MODEL` because the assertion is about a
/// behaviour a non-reasoning model would pass vacuously.
///
/// `None` when the variable is unset, so running the suite against an ordinary
/// chat GGUF skips these two rather than failing on a model it was never given.
fn thinking_model(assistant_prefix: Option<&str>) -> Option<LlamaCpp> {
    let path = std::env::var_os("PIPECRAB_LLAMA_THINKING_MODEL")?;
    let mut config = LlamaCppConfig::new(path).with_generation_defaults(
        NonZeroU32::new(192).expect("non-zero"),
        0.0,
        42,
    );
    if let Some(prefix) = assistant_prefix {
        config = config.with_assistant_prefix(prefix);
    }
    Some(LlamaCpp::load(config).expect("load model"))
}

/// Announce the skip: a test that asserts nothing still reports `ok`.
fn skipped(test: &str) {
    eprintln!("{test}: skipped, PIPECRAB_LLAMA_THINKING_MODEL is unset");
}

/// `llama_chat_apply_template` takes no template kwargs, so Qwen 3's
/// `enable_thinking=false` never fires and the model reasons into a transcript
/// a voice pipeline would speak aloud. Prefilling the assistant turn with the
/// block that branch emits applies it by hand.
#[test]
#[ignore = "requires PIPECRAB_LLAMA_THINKING_MODEL to point to a reasoning chat GGUF"]
fn an_assistant_prefix_suppresses_reasoning_without_swallowing_the_answer() {
    let Some(model) = thinking_model(Some(NO_THINK)) else {
        return skipped("an_assistant_prefix_suppresses_reasoning");
    };
    let (text, _) = turn(&model, &ask("What is the capital of France?"), &[]);

    for fragment in ["<think>", "</think>"] {
        assert!(
            !text.contains(fragment),
            "reasoning syntax {fragment:?} reached the transcript: {text:?}"
        );
    }
    // A prefix that suppressed the reply along with the reasoning would pass the
    // assertion above and be useless.
    assert!(
        text.to_lowercase().contains("paris"),
        "the answer did not survive the prefix: {text:?}"
    );
}

/// The prefix is prompt, not generation: it must not consume the turn's tool
/// call, which is the pairing a dispatching voice agent actually runs.
#[test]
#[ignore = "requires PIPECRAB_LLAMA_THINKING_MODEL to point to a reasoning chat GGUF"]
fn an_assistant_prefix_leaves_tool_calling_intact() {
    let Some(model) = thinking_model(Some(NO_THINK)) else {
        return skipped("an_assistant_prefix_leaves_tool_calling_intact");
    };
    let (text, calls) = turn(
        &model,
        &ask("Please kick off a background task to research sea otters."),
        &tools(),
    );

    assert_eq!(calls.len(), 1, "expected one tool call, got {calls:?}");
    assert_eq!(&*calls[0].name, "dispatch_task");
    for fragment in ["<think>", "</think>", "<tool_call>"] {
        assert!(
            !text.contains(fragment),
            "syntax {fragment:?} reached the transcript: {text:?}"
        );
    }
}
