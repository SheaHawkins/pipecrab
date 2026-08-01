//! One text turn and one tool turn against a real OpenAI-compatible host.
//!
//! Ignored by default: the workspace's own test run needs neither network nor
//! credentials. Point it at any host and run it explicitly:
//!
//! ```console
//! PIPECRAB_OPENAI_MODEL=gpt-5 \
//! PIPECRAB_OPENAI_API_KEY=sk-... \
//! cargo test -p pipecrab-lm-openai --test live -- --ignored --nocapture
//! ```
//!
//! `PIPECRAB_OPENAI_BASE_URL` is optional and defaults to OpenAI's endpoint; set
//! it to `https://openrouter.ai/api/v1` or `http://127.0.0.1:8080/v1` to exercise
//! another host. A local host that wants no key takes an empty
//! `PIPECRAB_OPENAI_API_KEY`.

use futures::StreamExt;
use pipecrab_lm::{Conversation, GenParams, LanguageModel, Message, ModelDelta, ToolDefinition};
use pipecrab_lm_openai::{OpenAi, OpenAiConfig};
use serde_json::json;
use url::Url;

/// Build the configured model, or explain which variable is missing.
fn model() -> OpenAi {
    let name = std::env::var("PIPECRAB_OPENAI_MODEL")
        .expect("PIPECRAB_OPENAI_MODEL must name the model to call");
    let api_key = std::env::var("PIPECRAB_OPENAI_API_KEY")
        .expect("PIPECRAB_OPENAI_API_KEY must be set (empty for a host wanting no key)");

    let mut config = OpenAiConfig::new(name, api_key);
    if let Ok(base_url) = std::env::var("PIPECRAB_OPENAI_BASE_URL") {
        config = config.with_base_url(Url::parse(&base_url).expect("a valid base URL"));
    }
    OpenAi::new(config).expect("a valid configuration")
}

fn dispatch_tool() -> ToolDefinition {
    ToolDefinition::new(
        "dispatch_task",
        "Hand a question or errand to a background agent that can browse the web. \
         Call this instead of saying you do not know or cannot check.",
        json!({
            "type": "object",
            "properties": {
                "task": { "type": "string", "description": "The work to perform." },
            },
            "required": ["task"],
        }),
    )
    .expect("valid tool")
}

/// Run one generation and collect what it produced.
async fn turn(user: &str, tools: &[ToolDefinition]) -> Vec<ModelDelta> {
    let conversation = Conversation {
        messages: vec![
            Message::system("You are a terse voice assistant."),
            Message::user(user),
        ],
    };
    let params = GenParams {
        max_tokens: Some(128),
        ..GenParams::default()
    };
    model()
        .generate(&conversation, &params, tools)
        .await
        .expect("the host accepted the request")
        .map(|item| item.expect("no delta failed"))
        .collect()
        .await
}

#[tokio::test]
#[ignore = "requires a reachable OpenAI-compatible host"]
async fn a_text_turn_streams_a_reply() {
    let deltas = turn("Say hello in five words.", &[]).await;
    let reply: String = deltas
        .iter()
        .filter_map(|delta| match delta {
            ModelDelta::Text(text) => Some(text.to_string()),
            ModelDelta::ToolCall(_) => None,
        })
        .collect();

    println!("reply: {reply}");
    assert!(!reply.trim().is_empty(), "the host produced no text");
    assert!(
        deltas.len() > 1,
        "the reply arrived in one lump, not streamed"
    );
}

#[tokio::test]
#[ignore = "requires a reachable OpenAI-compatible host"]
async fn a_tool_turn_produces_a_complete_call() {
    let deltas = turn(
        "What is the weather in Paris right now?",
        &[dispatch_tool()],
    )
    .await;
    let calls: Vec<_> = deltas
        .iter()
        .filter_map(|delta| match delta {
            ModelDelta::ToolCall(call) => Some(call),
            ModelDelta::Text(_) => None,
        })
        .collect();

    assert!(
        !calls.is_empty(),
        "the model reached for no tool: {deltas:?}"
    );
    for call in calls {
        println!("call: {} {}", call.name, call.arguments_json);
        assert_eq!(&*call.name, "dispatch_task");
        assert!(!call.id.is_empty(), "a call must carry an id");
        let arguments: serde_json::Value =
            serde_json::from_str(&call.arguments_json).expect("arguments are JSON");
        assert!(arguments.is_object(), "arguments must be an object");
    }
}
