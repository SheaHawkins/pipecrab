//! `OpenAi` against a mock Chat Completions host: what it puts on the wire,
//! what it makes of the response, and how it fails.
//!
//! The decoder's own protocol coverage is in the crate's unit tests; these pin
//! the parts that only exist once a real request crosses reqwest — headers, the
//! request body, status handling, and cancellation.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use pipecrab_lm::{
    Conversation, GenParams, LanguageModel, LmError, Message, ModelDelta, ToolDefinition,
};
use pipecrab_lm_openai::{OpenAi, OpenAiConfig};
use serde_json::{Value, json};
use url::Url;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const KEY: &str = "sk-test-secret";

/// A model pointed at `server`, with the given streaming mode.
fn model(server: &MockServer, streaming: bool) -> OpenAi {
    let base_url = Url::parse(&format!("{}/v1", server.uri())).expect("a mock URL");
    OpenAi::new(
        OpenAiConfig::new("test-model", KEY)
            .with_base_url(base_url)
            .with_streaming(streaming),
    )
    .expect("a valid config")
}

/// Answer `POST /v1/chat/completions` with `template`.
async fn mount(server: &MockServer, template: ResponseTemplate) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(template)
        .mount(server)
        .await;
}

/// An SSE response body.
fn sse(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(body.to_owned().into_bytes(), "text/event-stream")
}

fn conversation() -> Conversation {
    Conversation {
        messages: vec![Message::system("be brief"), Message::user("hello")],
    }
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "weather",
        "look up weather",
        json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
    )
    .expect("valid tool")
}

/// Drain a generation into its deltas.
async fn generate(
    model: &OpenAi,
    conversation: &Conversation,
    tools: &[ToolDefinition],
) -> Result<Vec<Result<ModelDelta, LmError>>, LmError> {
    let stream = model
        .generate(conversation, &GenParams::default(), tools)
        .await?;
    Ok(stream.collect().await)
}

fn text(items: &[Result<ModelDelta, LmError>]) -> String {
    items
        .iter()
        .filter_map(|item| match item {
            Ok(ModelDelta::Text(text)) => Some(text.to_string()),
            _ => None,
        })
        .collect()
}

fn tool_calls(items: &[Result<ModelDelta, LmError>]) -> Vec<(String, String)> {
    items
        .iter()
        .filter_map(|item| match item {
            Ok(ModelDelta::ToolCall(call)) => {
                Some((call.name.to_string(), call.arguments_json.to_string()))
            }
            _ => None,
        })
        .collect()
}

/// The body of the request the mock recorded.
async fn recorded_body(server: &MockServer, nth: usize) -> Value {
    let requests = server
        .received_requests()
        .await
        .expect("the mock server records requests");
    serde_json::from_slice(&requests[nth].body).expect("a JSON request body")
}

const HELLO_STREAM: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"lo \"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"there\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n",
);

const WEATHER_CALL_STREAM: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"weather\",\"arguments\":\"\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"paris\\\"}\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn a_text_turn_streams_its_deltas_in_order() {
    let server = MockServer::start().await;
    mount(&server, sse(HELLO_STREAM)).await;

    let items = generate(&model(&server, true), &conversation(), &[])
        .await
        .expect("a generation");

    assert_eq!(text(&items), "Hello there");
    assert!(items.iter().all(Result::is_ok), "{items:?}");
}

#[tokio::test]
async fn a_tool_turn_yields_one_complete_call_and_no_stray_text() {
    let server = MockServer::start().await;
    mount(&server, sse(WEATHER_CALL_STREAM)).await;

    let items = generate(&model(&server, true), &conversation(), &[weather_tool()])
        .await
        .expect("a generation");

    assert_eq!(
        tool_calls(&items),
        vec![("weather".to_string(), r#"{"city":"paris"}"#.to_string())]
    );
    assert_eq!(
        text(&items),
        "",
        "call syntax must never reach a transcript"
    );
}

#[tokio::test]
async fn the_request_carries_the_model_history_tools_and_bearer_token() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", format!("Bearer {KEY}").as_str()))
        .respond_with(sse(HELLO_STREAM))
        .mount(&server)
        .await;

    generate(&model(&server, true), &conversation(), &[weather_tool()])
        .await
        .expect("a generation");

    let body = recorded_body(&server, 0).await;
    assert_eq!(body["model"], json!("test-model"));
    assert_eq!(body["stream"], json!(true));
    assert_eq!(
        body["messages"],
        json!([
            { "role": "system", "content": "be brief" },
            { "role": "user", "content": "hello" },
        ])
    );
    assert_eq!(body["tools"][0]["function"]["name"], json!("weather"));
}

#[tokio::test]
async fn a_tool_result_and_an_event_reach_the_next_request_as_history() {
    let server = MockServer::start().await;
    mount(&server, sse(HELLO_STREAM)).await;

    let convo = Conversation {
        messages: vec![
            Message::user("what is the weather?"),
            Message::assistant_with_tool_calls(
                "",
                [pipecrab_lm::ToolCall {
                    id: Arc::from("call_a"),
                    name: Arc::from("weather"),
                    arguments_json: Arc::from(r#"{"city":"paris"}"#),
                }],
            ),
            Message::ToolResult {
                tool_call_id: Arc::from("call_a"),
                name: Arc::from("weather"),
                content: Arc::from("18C"),
            },
            Message::Event {
                source: Arc::from("hermes"),
                kind: Arc::from("completion"),
                content: Arc::from("the errand is done"),
            },
        ],
    };

    generate(&model(&server, true), &convo, &[weather_tool()])
        .await
        .expect("a generation");

    let messages = recorded_body(&server, 0).await["messages"].clone();
    assert_eq!(messages[1]["content"], Value::Null);
    assert_eq!(messages[1]["tool_calls"][0]["id"], json!("call_a"));
    assert_eq!(
        messages[2],
        json!({ "role": "tool", "tool_call_id": "call_a", "content": "18C" })
    );
    assert_eq!(
        messages[3],
        json!({ "role": "user", "content": "[hermes/completion] the errand is done" })
    );
}

#[tokio::test]
async fn an_unauthorized_response_reports_the_status_without_the_key() {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(401).set_body_string(r#"{"error":{"message":"invalid api key"}}"#),
    )
    .await;

    let error = generate(&model(&server, true), &conversation(), &[])
        .await
        .expect_err("a 401 fails the generation");

    let message = error.to_string();
    assert!(matches!(error, LmError::Engine(_)), "{error:?}");
    assert!(message.contains("401"), "{message}");
    assert!(message.contains("invalid api key"), "{message}");
    assert!(!message.contains(KEY), "{message}");
}

#[tokio::test]
async fn a_server_error_reports_its_status() {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(500).set_body_string("upstream exploded"),
    )
    .await;

    let error = generate(&model(&server, true), &conversation(), &[])
        .await
        .expect_err("a 500 fails the generation");

    let message = error.to_string();
    assert!(message.contains("500"), "{message}");
    assert!(message.contains("upstream exploded"), "{message}");
}

#[tokio::test]
async fn a_body_that_ends_mid_event_fails_the_stream() {
    let server = MockServer::start().await;
    mount(&server, sse("data: {\"choices\":[{\"delta\":{\"content\"")).await;

    let items = generate(&model(&server, true), &conversation(), &[])
        .await
        .expect("the request itself succeeds");

    assert!(
        matches!(items.as_slice(), [Err(LmError::ProviderStream(_))]),
        "{items:?}"
    );
}

#[tokio::test]
async fn non_streaming_mode_produces_the_same_deltas() {
    let server = MockServer::start().await;
    mount(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": "Hello there",
                    "tool_calls": [{
                        "id": "call_a",
                        "type": "function",
                        "function": { "name": "weather", "arguments": "{\"city\":\"paris\"}" },
                    }],
                },
            }],
        })),
    )
    .await;

    let items = generate(&model(&server, false), &conversation(), &[weather_tool()])
        .await
        .expect("a generation");

    assert_eq!(text(&items), "Hello there");
    assert_eq!(
        tool_calls(&items),
        vec![("weather".to_string(), r#"{"city":"paris"}"#.to_string())]
    );
    assert!(
        recorded_body(&server, 0).await.get("stream").is_none(),
        "a blocking request does not ask for a stream"
    );
}

#[tokio::test]
async fn cancel_mid_stream_ends_the_turn_without_an_error() {
    let server = MockServer::start().await;
    mount(&server, sse(HELLO_STREAM)).await;

    let model = model(&server, true);
    let mut stream = model
        .generate(&conversation(), &GenParams::default(), &[])
        .await
        .expect("a generation");

    assert!(stream.next().await.is_some(), "the reply starts");
    model.cancel();
    assert!(
        stream.next().await.is_none(),
        "a cancelled turn ends, it does not fail"
    );
}

#[tokio::test]
async fn cancel_before_the_response_arrives_yields_nothing() {
    let server = MockServer::start().await;
    mount(
        &server,
        sse(HELLO_STREAM).set_delay(Duration::from_millis(500)),
    )
    .await;

    let model = model(&server, true);
    let pending = {
        let model = model.clone();
        tokio::spawn(async move {
            model
                .generate(&conversation(), &GenParams::default(), &[])
                .await
        })
    };

    // Land the cancel while the request is still in flight.
    tokio::time::sleep(Duration::from_millis(50)).await;
    model.cancel();

    let mut stream = pending
        .await
        .expect("the task joins")
        .expect("a generation");
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn an_empty_key_sends_no_authorization_header() {
    let server = MockServer::start().await;
    mount(&server, sse(HELLO_STREAM)).await;

    let base_url = Url::parse(&format!("{}/v1", server.uri())).expect("a mock URL");
    let model = OpenAi::new(OpenAiConfig::new("local-model", "").with_base_url(base_url))
        .expect("a valid config");
    generate(&model, &conversation(), &[])
        .await
        .expect("a generation");

    let requests = server.received_requests().await.expect("recorded requests");
    assert!(requests[0].headers.get("authorization").is_none());
}

#[tokio::test]
async fn extra_headers_and_body_fields_reach_the_host() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("http-referer", "https://example.com"))
        .respond_with(sse(HELLO_STREAM))
        .mount(&server)
        .await;

    let base_url = Url::parse(&format!("{}/v1", server.uri())).expect("a mock URL");
    let mut extra = serde_json::Map::new();
    extra.insert("provider".into(), json!({ "order": ["together"] }));
    let model = OpenAi::new(
        OpenAiConfig::new("test-model", KEY)
            .with_base_url(base_url)
            .with_header("HTTP-Referer", "https://example.com")
            .with_extra_body(extra),
    )
    .expect("a valid config");

    generate(&model, &conversation(), &[])
        .await
        .expect("a generation");

    assert_eq!(
        recorded_body(&server, 0).await["provider"],
        json!({ "order": ["together"] })
    );
}

#[tokio::test]
async fn session_state_is_empty_and_refuses_a_foreign_blob() {
    let server = MockServer::start().await;
    let model = model(&server, true);

    assert!(model.save_state().await.expect("a blob").is_empty());
    assert!(model.load_state(&[]).await.is_ok());
    assert!(matches!(
        model.load_state(b"something").await,
        Err(LmError::Engine(_))
    ));
}
