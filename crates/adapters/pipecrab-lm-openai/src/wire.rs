//! The Chat Completions wire format: the request body this adapter posts, and
//! the response shapes it reads back.
//!
//! Pure translation — no HTTP, no streaming state — so the whole format is
//! testable without a server.

use pipecrab_lm::{Conversation, GenParams, LmError, Message, ToolCall, ToolDefinition};
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Build the `POST /chat/completions` body.
pub(crate) fn build_request(
    model: &str,
    conversation: &Conversation,
    params: &GenParams,
    tools: &[ToolDefinition],
    streaming: bool,
    extra_body: &Map<String, Value>,
) -> Result<Value, LmError> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), json!(render_messages(conversation)));
    if streaming {
        body.insert("stream".into(), json!(true));
    }
    // Omitted rather than sent empty: a host that rejects an empty array stays
    // happy, and a turn with no tools is indistinguishable from a tool-free
    // adapter. `tool_choice` is never sent — every host defaults to auto.
    if !tools.is_empty() {
        body.insert("tools".into(), json!(render_tools(tools)));
    }
    if let Some(max_tokens) = params.max_tokens {
        body.insert("max_tokens".into(), json!(max_tokens));
    }
    if let Some(temperature) = params.temperature {
        body.insert("temperature".into(), decimal(temperature));
    }
    if let Some(grammar) = &params.grammar {
        body.insert("response_format".into(), response_format(grammar)?);
    }
    // Last, so a host-specific field can also correct one set above.
    for (key, value) in extra_body {
        body.insert(key.clone(), value.clone());
    }
    Ok(Value::Object(body))
}

/// Widen an `f32` knob to JSON's `f64` through its shortest round-tripping
/// decimal, so `0.2` reaches the host as `0.2` rather than the nearest double to
/// the `f32`, `0.20000000298023224`. Both parse to the same sampling
/// temperature; only one is readable in a request log.
fn decimal(value: f32) -> Value {
    value
        .to_string()
        .parse::<f64>()
        .map(Value::from)
        .unwrap_or_else(|_| json!(value))
}

/// Read [`GenParams::grammar`] as a JSON Schema document and wrap it in the
/// structured-output request field. Text that is not JSON is an error rather
/// than a silently dropped constraint.
fn response_format(grammar: &str) -> Result<Value, LmError> {
    let schema: Value = serde_json::from_str(grammar).map_err(|error| {
        LmError::Engine(format!(
            "GenParams::grammar must be a JSON Schema document: {error}"
        ))
    })?;
    Ok(json!({
        "type": "json_schema",
        "json_schema": { "name": "response", "schema": schema },
    }))
}

/// Render conversation history as OpenAI `messages`.
fn render_messages(conversation: &Conversation) -> Vec<Value> {
    conversation
        .messages
        .iter()
        .map(|message| match message {
            Message::System { content } => json!({ "role": "system", "content": &**content }),
            Message::User { content } => json!({ "role": "user", "content": &**content }),
            Message::Assistant {
                content,
                tool_calls,
            } if tool_calls.is_empty() => {
                json!({ "role": "assistant", "content": &**content })
            }
            Message::Assistant {
                content,
                tool_calls,
            } => json!({
                "role": "assistant",
                // Null, not "": hosts reject an assistant turn carrying calls
                // alongside empty content.
                "content": if content.is_empty() { Value::Null } else { json!(&**content) },
                "tool_calls": tool_calls.iter().map(render_tool_call).collect::<Vec<_>>(),
            }),
            // The tool's name is dropped: it is deprecated on OpenAI's tool
            // message and rejected outright by stricter hosts. `tool_call_id` is
            // the correlator that matters.
            Message::ToolResult {
                tool_call_id,
                content,
                ..
            } => json!({
                "role": "tool",
                "tool_call_id": &**tool_call_id,
                "content": &**content,
            }),
            // No `event` role exists. A labelled user turn matches what the
            // llama.cpp adapter renders, so history reads the same either side.
            Message::Event {
                source,
                kind,
                content,
            } => json!({
                "role": "user",
                "content": format!("[{source}/{kind}] {content}"),
            }),
        })
        .collect()
}

/// Replay one assistant tool call. `arguments` is already validated JSON *text*,
/// which is exactly what the field holds — no re-encoding.
fn render_tool_call(call: &ToolCall) -> Value {
    json!({
        "id": &*call.id,
        "type": "function",
        "function": { "name": &*call.name, "arguments": &*call.arguments_json },
    })
}

/// Render the stage's tools as OpenAI function declarations.
fn render_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": &*tool.name,
                    "description": &*tool.description,
                    "parameters": tool.parameters,
                },
            })
        })
        .collect()
}

/// One `data:` event of a streamed completion.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ChatChunk {
    #[serde(default)]
    pub(crate) choices: Vec<ChunkChoice>,
    /// Some hosts report a mid-stream failure as an ordinary event.
    #[serde(default)]
    pub(crate) error: Option<Value>,
}

/// One choice of a streamed chunk. Only the first is read: `n > 1` is out of
/// scope, and a single reply is what a conversation loop consumes.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ChunkChoice {
    #[serde(default)]
    pub(crate) delta: Delta,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}

/// The incremental payload of one chunk.
///
/// `reasoning` and `reasoning_content` are deliberately absent: serde drops
/// unknown fields, and chain-of-thought must never reach a transcript the TTS
/// stage would speak aloud.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct Delta {
    #[serde(default)]
    pub(crate) content: Option<String>,
    #[serde(default)]
    pub(crate) tool_calls: Vec<ToolCallFragment>,
}

/// A slice of one tool call. Every field is optional because a host splits a
/// call across chunks: id and name first, then `arguments` in pieces.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ToolCallFragment {
    #[serde(default)]
    pub(crate) index: Option<u32>,
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    pub(crate) function: Option<FunctionFragment>,
}

/// The function half of a [`ToolCallFragment`].
#[derive(Debug, Default, Deserialize)]
pub(crate) struct FunctionFragment {
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) arguments: Option<String>,
}

/// A whole non-streamed completion.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ChatResponse {
    #[serde(default)]
    pub(crate) choices: Vec<ResponseChoice>,
    #[serde(default)]
    pub(crate) error: Option<Value>,
}

/// One choice of a non-streamed completion.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ResponseChoice {
    #[serde(default)]
    pub(crate) message: ResponseMessage,
}

/// The assembled assistant turn a non-streamed completion carries.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ResponseMessage {
    #[serde(default)]
    pub(crate) content: Option<String>,
    #[serde(default)]
    pub(crate) tool_calls: Vec<ToolCallFragment>,
}

/// Read a message out of an `error` field: a bare string, or an object with a
/// string `message`. Anything else is reported verbatim.
pub(crate) fn error_detail(value: &Value) -> String {
    match value {
        Value::String(message) => message.clone(),
        Value::Object(map) => map
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| value.to_string()),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn conversation(messages: Vec<Message>) -> Conversation {
        Conversation { messages }
    }

    fn weather_tool() -> ToolDefinition {
        ToolDefinition::new(
            "weather",
            "look up weather",
            json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        )
        .expect("valid tool")
    }

    fn request(conversation: &Conversation, tools: &[ToolDefinition]) -> Value {
        build_request(
            "gpt-5",
            conversation,
            &GenParams::default(),
            tools,
            true,
            &Map::new(),
        )
        .expect("valid request")
    }

    #[test]
    fn system_and_user_turns_render_by_role() {
        let convo = conversation(vec![Message::system("be brief"), Message::user("hello")]);
        assert_eq!(
            request(&convo, &[])["messages"],
            json!([
                { "role": "system", "content": "be brief" },
                { "role": "user", "content": "hello" },
            ])
        );
    }

    #[test]
    fn a_plain_assistant_turn_keeps_string_content_and_omits_tool_calls() {
        let convo = conversation(vec![Message::assistant("sure")]);
        assert_eq!(
            request(&convo, &[])["messages"],
            json!([{ "role": "assistant", "content": "sure" }])
        );
    }

    #[test]
    fn a_tool_call_only_assistant_turn_sends_null_content() {
        let call = ToolCall {
            id: Arc::from("call-1"),
            name: Arc::from("weather"),
            arguments_json: Arc::from(r#"{"city":"paris"}"#),
        };
        let convo = conversation(vec![Message::assistant_with_tool_calls("", [call])]);
        assert_eq!(
            request(&convo, &[])["messages"],
            json!([{
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": { "name": "weather", "arguments": r#"{"city":"paris"}"# },
                }],
            }])
        );
    }

    #[test]
    fn an_assistant_turn_with_both_text_and_calls_keeps_the_text() {
        let call = ToolCall {
            id: Arc::from("call-1"),
            name: Arc::from("weather"),
            arguments_json: Arc::from("{}"),
        };
        let convo = conversation(vec![Message::assistant_with_tool_calls(
            "one moment",
            [call],
        )]);
        assert_eq!(
            request(&convo, &[])["messages"][0]["content"],
            json!("one moment")
        );
    }

    #[test]
    fn a_tool_result_renders_as_a_tool_message_without_its_name() {
        let convo = conversation(vec![Message::ToolResult {
            tool_call_id: Arc::from("call-1"),
            name: Arc::from("weather"),
            content: Arc::from("18C"),
        }]);
        let messages = request(&convo, &[])["messages"].clone();
        assert_eq!(
            messages,
            json!([{ "role": "tool", "tool_call_id": "call-1", "content": "18C" }])
        );
        assert!(messages[0].get("name").is_none(), "{messages}");
    }

    #[test]
    fn an_event_renders_as_a_labelled_user_turn() {
        let convo = conversation(vec![Message::Event {
            source: Arc::from("hermes"),
            kind: Arc::from("completion"),
            content: Arc::from("the errand is done"),
        }]);
        assert_eq!(
            request(&convo, &[])["messages"],
            json!([{ "role": "user", "content": "[hermes/completion] the errand is done" }])
        );
    }

    #[test]
    fn tools_render_as_function_declarations() {
        let convo = conversation(vec![Message::user("weather?")]);
        assert_eq!(
            request(&convo, &[weather_tool()])["tools"],
            json!([{
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "look up weather",
                    "parameters": { "type": "object", "properties": { "city": { "type": "string" } } },
                },
            }])
        );
    }

    #[test]
    fn an_empty_tool_set_omits_the_field() {
        let convo = conversation(vec![Message::user("hi")]);
        assert!(request(&convo, &[]).get("tools").is_none());
    }

    #[test]
    fn generation_params_are_sent_only_when_set() {
        let convo = conversation(vec![Message::user("hi")]);
        let bare = request(&convo, &[]);
        assert!(bare.get("max_tokens").is_none());
        assert!(bare.get("temperature").is_none());
        assert!(bare.get("response_format").is_none());

        let params = GenParams {
            max_tokens: Some(64),
            temperature: Some(0.2),
            grammar: None,
        };
        let set = build_request("gpt-5", &convo, &params, &[], true, &Map::new()).unwrap();
        assert_eq!(set["max_tokens"], json!(64));
        assert_eq!(set["temperature"], json!(0.2));
    }

    #[test]
    fn an_f32_knob_serializes_as_the_decimal_the_caller_wrote() {
        assert_eq!(decimal(0.2), json!(0.2));
        assert_eq!(decimal(0.7), json!(0.7));
        assert_eq!(decimal(1.0), json!(1.0));
        assert_eq!(serde_json::to_string(&decimal(0.2)).unwrap(), "0.2");
    }

    #[test]
    fn a_grammar_becomes_a_json_schema_response_format() {
        let convo = conversation(vec![Message::user("hi")]);
        let params = GenParams {
            grammar: Some(Arc::from(r#"{"type":"object"}"#)),
            ..GenParams::default()
        };
        let body = build_request("gpt-5", &convo, &params, &[], true, &Map::new()).unwrap();
        assert_eq!(
            body["response_format"],
            json!({
                "type": "json_schema",
                "json_schema": { "name": "response", "schema": { "type": "object" } },
            })
        );
    }

    #[test]
    fn a_grammar_that_is_not_json_fails_the_request() {
        let convo = conversation(vec![Message::user("hi")]);
        let params = GenParams {
            grammar: Some(Arc::from("root ::= \"yes\"")),
            ..GenParams::default()
        };
        let error = build_request("gpt-5", &convo, &params, &[], true, &Map::new()).unwrap_err();
        assert!(matches!(error, LmError::Engine(_)), "{error:?}");
    }

    #[test]
    fn non_streaming_omits_the_stream_field() {
        let convo = conversation(vec![Message::user("hi")]);
        let body = build_request(
            "gpt-5",
            &convo,
            &GenParams::default(),
            &[],
            false,
            &Map::new(),
        )
        .unwrap();
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn extra_body_merges_into_the_root_and_can_correct_a_field() {
        let convo = conversation(vec![Message::user("hi")]);
        let mut extra = Map::new();
        extra.insert("provider".into(), json!({ "order": ["together"] }));
        extra.insert("model".into(), json!("overridden"));

        let body =
            build_request("gpt-5", &convo, &GenParams::default(), &[], true, &extra).unwrap();
        assert_eq!(body["provider"], json!({ "order": ["together"] }));
        assert_eq!(body["model"], json!("overridden"));
    }

    #[test]
    fn error_detail_reads_a_string_an_object_and_falls_back() {
        assert_eq!(error_detail(&json!("boom")), "boom");
        assert_eq!(
            error_detail(&json!({ "message": "rate limited", "code": 429 })),
            "rate limited"
        );
        assert_eq!(error_detail(&json!({ "code": 429 })), r#"{"code":429}"#);
    }
}
