//! `ZeroclawLm` against an in-process mock daemon: a unix-socket JSON-RPC
//! server scripted per test. The mock auto-answers the `initialize` +
//! `session/new` handshake (forwarding both on a separate channel so
//! reconnect tests can assert identity echo) and hands every other request
//! to the test, which answers over a raw line channel.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use pipecrab_lm::{Conversation, GenParams, LanguageModel, LmError, Message, ModelDelta};
use pipecrab_lm_zeroclaw::{ZeroclawLmConfig, connect};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;

const WAIT: Duration = Duration::from_secs(5);
/// Long enough for the adapter to have acted, short enough not to drag.
const QUIET: Duration = Duration::from_millis(200);

#[derive(Debug)]
struct Request {
    method: String,
    id: u64,
    params: Value,
}

struct MockDaemon {
    /// Non-handshake requests (session/prompt, session/cancel, …).
    requests: mpsc::UnboundedReceiver<Request>,
    /// Handshake requests (initialize, session/new), in arrival order.
    handshakes: mpsc::UnboundedReceiver<Request>,
    /// Raw lines to write to the currently active connection.
    out: mpsc::UnboundedSender<String>,
    /// Force-close the active connection.
    close: mpsc::UnboundedSender<()>,
    socket: PathBuf,
    workspace: PathBuf,
    _dir: tempfile::TempDir,
}

impl MockDaemon {
    fn spawn() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let socket = dir.path().join("d.sock");
        let workspace = dir.path().join("ws");
        std::fs::create_dir_all(&workspace).expect("workspace dir");

        let (requests_tx, requests) = mpsc::unbounded_channel();
        let (handshakes_tx, handshakes) = mpsc::unbounded_channel();
        let (out_tx, out_rx) = mpsc::unbounded_channel::<String>();
        let (close_tx, close_rx) = mpsc::unbounded_channel::<()>();

        let listener = {
            let _guard = ();
            UnixListener::bind(&socket).expect("bind mock daemon socket")
        };
        let workspace_wire = workspace.to_str().expect("utf-8 workspace").to_owned();
        let out_tx_server = out_tx.clone();
        tokio::spawn(serve(
            listener,
            workspace_wire,
            requests_tx,
            handshakes_tx,
            out_tx_server,
            Arc::new(tokio::sync::Mutex::new(out_rx)),
            Arc::new(tokio::sync::Mutex::new(close_rx)),
        ));

        Self {
            requests,
            handshakes,
            out: out_tx,
            close: close_tx,
            socket,
            workspace,
            _dir: dir,
        }
    }

    fn config(&self, agent: &str) -> ZeroclawLmConfig {
        ZeroclawLmConfig::new(agent)
            .with_socket_path(&self.socket)
            .with_session_id("sess-test")
    }

    async fn next_request(&mut self) -> Request {
        tokio::time::timeout(WAIT, self.requests.recv())
            .await
            .expect("timed out waiting for a request")
            .expect("mock daemon request channel closed")
    }

    async fn next_handshake(&mut self) -> Request {
        tokio::time::timeout(WAIT, self.handshakes.recv())
            .await
            .expect("timed out waiting for a handshake request")
            .expect("mock daemon handshake channel closed")
    }

    async fn assert_quiet(&mut self) {
        let outcome = tokio::time::timeout(QUIET, self.requests.recv()).await;
        assert!(
            outcome.is_err(),
            "expected no request, got {:?}",
            outcome.unwrap()
        );
    }

    fn respond_ok(&self, id: u64, result: Value) {
        let line = json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string();
        self.out.send(line).expect("mock out channel");
    }

    fn respond_err(&self, id: u64, code: i64, message: &str) {
        let line = json!({
            "jsonrpc": "2.0", "id": id,
            "error": { "code": code, "message": message },
        })
        .to_string();
        self.out.send(line).expect("mock out channel");
    }

    fn update(&self, params: Value) {
        let line = json!({
            "jsonrpc": "2.0", "method": "session/update", "params": params,
        })
        .to_string();
        self.out.send(line).expect("mock out channel");
    }

    fn chunk(&self, session: &str, text: &str) {
        self.update(json!({
            "type": "agent_message_chunk", "session_id": session, "text": text,
        }));
    }

    fn turn_complete(&self, session: &str, outcome: &str, content: &str) {
        self.update(json!({
            "type": "turn_complete", "session_id": session,
            "outcome": outcome, "content": content,
        }));
    }

    fn drop_connection(&self) {
        self.close.send(()).expect("mock close channel");
    }
}

#[allow(clippy::too_many_arguments)]
async fn serve(
    listener: UnixListener,
    workspace: String,
    requests: mpsc::UnboundedSender<Request>,
    handshakes: mpsc::UnboundedSender<Request>,
    out_tx: mpsc::UnboundedSender<String>,
    out_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<String>>>,
    close_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<()>>>,
) {
    while let Ok((stream, _)) = listener.accept().await {
        serve_connection(
            stream,
            &workspace,
            &requests,
            &handshakes,
            &out_tx,
            &out_rx,
            &close_rx,
        )
        .await;
    }
}

async fn serve_connection(
    stream: UnixStream,
    workspace: &str,
    requests: &mpsc::UnboundedSender<Request>,
    handshakes: &mpsc::UnboundedSender<Request>,
    out_tx: &mpsc::UnboundedSender<String>,
    out_rx: &Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<String>>>,
    close_rx: &Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<()>>>,
) {
    let (read_half, mut write_half) = stream.into_split();
    let out_rx_task = Arc::clone(out_rx);
    let writer = tokio::spawn(async move {
        loop {
            let line = { out_rx_task.lock().await.recv().await };
            let Some(mut line) = line else { break };
            line.push('\n');
            if write_half.write_all(line.as_bytes()).await.is_err() {
                break;
            }
        }
    });

    let mut lines = BufReader::new(read_half).lines();
    loop {
        let closed = {
            let mut close = close_rx.lock().await;
            tokio::select! {
                signal = close.recv() => signal.is_some(),
                line = lines.next_line() => {
                    match line {
                        Ok(Some(line)) => {
                            handle_line(&line, workspace, requests, handshakes, out_tx);
                            continue;
                        }
                        _ => true,
                    }
                }
            }
        };
        if closed {
            break;
        }
    }
    writer.abort();
}

fn handle_line(
    line: &str,
    workspace: &str,
    requests: &mpsc::UnboundedSender<Request>,
    handshakes: &mpsc::UnboundedSender<Request>,
    out_tx: &mpsc::UnboundedSender<String>,
) {
    let value: Value = serde_json::from_str(line).expect("mock daemon received invalid JSON");
    let method = value["method"].as_str().expect("request method").to_owned();
    let id = value["id"].as_u64().expect("request id");
    let params = value["params"].clone();
    let request = Request {
        method: method.clone(),
        id,
        params: params.clone(),
    };

    match method.as_str() {
        "initialize" => {
            let _ = handshakes.send(request);
            let line = json!({
                "jsonrpc": "2.0", "id": id,
                "result": {
                    "protocol_version": 1, "server_version": "mock",
                    "tui_id": "tui-1", "tui_sig": "sig-1",
                },
            })
            .to_string();
            let _ = out_tx.send(line);
        }
        "session/new" => {
            let session_id = params["session_id"].as_str().unwrap_or("sess-test");
            let _ = handshakes.send(request);
            let line = json!({
                "jsonrpc": "2.0", "id": id,
                "result": {
                    "session_id": session_id,
                    "agent_alias": params["agent_alias"],
                    "message_count": 0,
                    "workspace_dir": workspace,
                },
            })
            .to_string();
            let _ = out_tx.send(line);
        }
        _ => {
            let _ = requests.send(request);
        }
    }
}

fn user_convo(text: &str) -> Conversation {
    Conversation {
        messages: vec![Message::system("sys"), Message::user(text)],
    }
}

async fn drain(stream: &mut pipecrab_lm::ModelStream) -> Vec<Result<ModelDelta, LmError>> {
    let mut items = Vec::new();
    loop {
        match tokio::time::timeout(WAIT, stream.next()).await {
            Ok(Some(item)) => items.push(item),
            Ok(None) => return items,
            Err(_) => panic!("timed out draining the model stream; got {items:?}"),
        }
    }
}

async fn next_delta(stream: &mut pipecrab_lm::ModelStream) -> Result<ModelDelta, LmError> {
    tokio::time::timeout(WAIT, stream.next())
        .await
        .expect("timed out waiting for a delta")
        .expect("stream ended while a delta was expected")
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_bootstraps_and_surfaces_session_metadata() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let init = mock.next_handshake().await;
    assert_eq!(init.method, "initialize");
    assert_eq!(init.params["protocol_version"], 1);

    let new = mock.next_handshake().await;
    assert_eq!(new.method, "session/new");
    assert_eq!(new.params["agent_alias"], "voice");
    assert_eq!(new.params["chat_mode"], "chat");

    assert_eq!(lm.session_id(), "sess-test");
    assert_eq!(lm.workspace_dir(), mock.workspace.as_path());
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_fails_on_unknown_agent() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("d.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    // A daemon that accepts, answers initialize, and rejects session/new.
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let (read_half, mut write_half) = stream.into_split();
        let mut lines = BufReader::new(read_half).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let value: Value = serde_json::from_str(&line).unwrap();
            let id = value["id"].as_u64().unwrap();
            let reply = match value["method"].as_str().unwrap() {
                "initialize" => json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": { "protocol_version": 1, "server_version": "mock" },
                }),
                _ => json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": { "code": -32602, "message": "unknown agent alias: nope" },
                }),
            };
            let mut line = reply.to_string();
            line.push('\n');
            write_half.write_all(line.as_bytes()).await.unwrap();
        }
    });

    let config = ZeroclawLmConfig::new("nope").with_socket_path(&socket);
    let Err(error) = connect(config).await else {
        panic!("session/new must fail");
    };
    let message = error.to_string();
    assert!(
        message.contains("unknown agent alias"),
        "unexpected error: {message}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn streams_chunks_and_ignores_non_speech_updates() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream = lm
        .generate(&user_convo("hello there"), &GenParams::default(), &[])
        .await
        .expect("generate");

    let prompt = mock.next_request().await;
    assert_eq!(prompt.method, "session/prompt");
    assert_eq!(prompt.params["session_id"], "sess-test");
    assert_eq!(prompt.params["prompt"], "hello there");

    // A tool_result BEFORE any text yields neither a delta nor a gap.
    mock.update(json!({ "type": "tool_result", "session_id": "sess-test",
        "tool_call_id": "c1", "name": "shell", "raw_output": "ok" }));
    mock.chunk("sess-test", "Hel");
    mock.update(json!({ "type": "agent_thought_chunk", "session_id": "sess-test", "text": "hmm" }));
    mock.update(json!({ "type": "context_usage", "session_id": "sess-test", "input_tokens": 10 }));
    mock.update(json!({ "type": "plan", "session_id": "sess-test", "entries": [] }));
    mock.chunk("sess-test", "lo");
    mock.turn_complete("sess-test", "completed", "Hello");
    mock.respond_ok(
        prompt.id,
        json!({ "session_id": "sess-test", "stop_reason": "end", "content": "Hello" }),
    );

    let items = drain(&mut stream).await;
    assert_eq!(
        items,
        vec![
            Ok(ModelDelta::Text("Hel".into())),
            Ok(ModelDelta::Text("lo".into())),
        ],
        "only speech chunks may become deltas, and the terminal content must \
         not be re-emitted after streamed chunks"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn foreign_session_updates_are_ignored() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream = lm
        .generate(&user_convo("hi"), &GenParams::default(), &[])
        .await
        .expect("generate");
    let prompt = mock.next_request().await;

    mock.chunk("someone-else", "intruder");
    mock.turn_complete("someone-else", "completed", "intruder");
    mock.chunk("sess-test", "mine");
    mock.turn_complete("sess-test", "completed", "mine");
    mock.respond_ok(prompt.id, json!({ "content": "mine" }));

    let items = drain(&mut stream).await;
    assert_eq!(items, vec![Ok(ModelDelta::Text("mine".into()))]);
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_calls_surface_with_object_arguments_only() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream = lm
        .generate(&user_convo("check the weather"), &GenParams::default(), &[])
        .await
        .expect("generate");
    let prompt = mock.next_request().await;

    mock.update(json!({
        "type": "tool_call", "session_id": "sess-test",
        "tool_call_id": "call-1", "name": "http_request",
        "raw_input": { "url": "https://example.com" },
    }));
    // Non-object arguments are not surfaceable.
    mock.update(json!({
        "type": "tool_call", "session_id": "sess-test",
        "tool_call_id": "call-2", "name": "shell", "raw_input": "ls",
    }));
    mock.turn_complete("sess-test", "completed", "");
    mock.respond_ok(prompt.id, json!({ "content": "" }));

    let items = drain(&mut stream).await;
    assert_eq!(items.len(), 1, "got {items:?}");
    let Ok(ModelDelta::ToolCall(call)) = &items[0] else {
        panic!("expected a tool-call delta, got {items:?}");
    };
    assert_eq!(&*call.id, "call-1");
    assert_eq!(&*call.name, "http_request");
}

#[tokio::test(flavor = "multi_thread")]
async fn surfacing_can_be_disabled() {
    let mut mock = MockDaemon::spawn();
    let config = mock.config("voice").with_surface_tool_calls(false);
    let (lm, _source) = connect(config).await.expect("connect");

    let mut stream = lm
        .generate(&user_convo("check"), &GenParams::default(), &[])
        .await
        .expect("generate");
    let prompt = mock.next_request().await;

    mock.update(json!({
        "type": "tool_call", "session_id": "sess-test",
        "tool_call_id": "call-1", "name": "shell", "raw_input": {},
    }));
    mock.turn_complete("sess-test", "completed", "done");
    mock.respond_ok(prompt.id, json!({ "content": "done" }));

    let items = drain(&mut stream).await;
    assert_eq!(items, vec![Ok(ModelDelta::Text("done".into()))]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tool_boundary_injects_a_speakable_gap() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream = lm
        .generate(&user_convo("weather in denver"), &GenParams::default(), &[])
        .await
        .expect("generate");
    let prompt = mock.next_request().await;

    // Iteration 1's text ends without whitespace, then the tool round, then
    // iteration 2's text — the daemon concatenates these with no separator,
    // which would deny the sentence chunker its terminator-then-whitespace
    // boundary and hold ALL speech until the turn's final flush.
    mock.chunk("sess-test", "I'll get that.");
    mock.update(json!({
        "type": "tool_call", "session_id": "sess-test",
        "tool_call_id": "call-1", "name": "delegate",
        "raw_input": { "agent": "research", "background": true },
    }));
    mock.update(json!({ "type": "tool_result", "session_id": "sess-test",
        "tool_call_id": "call-1", "name": "delegate", "raw_output": "task_id: t1" }));
    mock.chunk("sess-test", "I've started it.");
    mock.turn_complete("sess-test", "completed", "I'll get that. I've started it.");
    mock.respond_ok(prompt.id, json!({ "content": "" }));

    let items = drain(&mut stream).await;
    assert_eq!(items.len(), 4, "got {items:?}");
    assert_eq!(items[0], Ok(ModelDelta::Text("I'll get that.".into())));
    // The gap arrives AT the boundary — before the tool call — so the
    // finished sentence starts synthesizing while the tool runs.
    assert_eq!(items[1], Ok(ModelDelta::Text(" ".into())));
    assert!(
        matches!(&items[2], Ok(ModelDelta::ToolCall(call)) if &*call.name == "delegate"),
        "got {items:?}"
    );
    assert_eq!(items[3], Ok(ModelDelta::Text("I've started it.".into())));
}

#[tokio::test(flavor = "multi_thread")]
async fn no_gap_when_the_text_already_ends_in_whitespace() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream = lm
        .generate(&user_convo("hi"), &GenParams::default(), &[])
        .await
        .expect("generate");
    let prompt = mock.next_request().await;

    mock.chunk("sess-test", "One moment. ");
    mock.update(json!({
        "type": "tool_call", "session_id": "sess-test",
        "tool_call_id": "call-1", "name": "delegate", "raw_input": {},
    }));
    mock.chunk("sess-test", "Started.");
    mock.turn_complete("sess-test", "completed", "");
    mock.respond_ok(prompt.id, json!({ "content": "" }));

    let items = drain(&mut stream).await;
    assert_eq!(
        items,
        vec![
            Ok(ModelDelta::Text("One moment. ".into())),
            Ok(ModelDelta::tool_call("call-1", "delegate", json!({})).unwrap()),
            Ok(ModelDelta::Text("Started.".into())),
        ],
        "a trailing space already provides the boundary; no gap owed"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn non_streamed_turn_emits_terminal_content_once() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream = lm
        .generate(&user_convo("hi"), &GenParams::default(), &[])
        .await
        .expect("generate");
    let prompt = mock.next_request().await;

    mock.turn_complete("sess-test", "completed", "whole reply at once");
    mock.respond_ok(prompt.id, json!({ "content": "whole reply at once" }));

    let items = drain(&mut stream).await;
    assert_eq!(
        items,
        vec![Ok(ModelDelta::Text("whole reply at once".into()))]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_turn_surfaces_a_recoverable_error() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream = lm
        .generate(&user_convo("hi"), &GenParams::default(), &[])
        .await
        .expect("generate");
    let prompt = mock.next_request().await;

    mock.turn_complete("sess-test", "failed", "provider exploded");
    mock.respond_ok(prompt.id, json!({ "content": "" }));

    let items = drain(&mut stream).await;
    assert_eq!(items.len(), 1);
    assert!(
        matches!(&items[0], Err(LmError::Engine(m)) if m.contains("provider exploded")),
        "got {items:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn rejected_prompt_is_the_turn_terminal() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream = lm
        .generate(&user_convo("hi"), &GenParams::default(), &[])
        .await
        .expect("generate");
    let prompt = mock.next_request().await;

    mock.respond_err(prompt.id, -32602, "empty user message");

    let items = drain(&mut stream).await;
    assert_eq!(items.len(), 1);
    assert!(
        matches!(&items[0], Err(LmError::Engine(m)) if m.contains("empty user message")),
        "got {items:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn tools_are_rejected_and_gen_params_ignored() {
    let mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let tool = pipecrab_lm::ToolDefinition::new("t", "d", json!({})).unwrap();
    let Err(error) = lm
        .generate(&user_convo("hi"), &GenParams::default(), &[tool])
        .await
    else {
        panic!("tools must be rejected");
    };
    assert!(matches!(error, LmError::Engine(m) if m.contains("agent profile")));
}

#[tokio::test(flavor = "multi_thread")]
async fn whitespace_input_yields_an_empty_stream_without_a_prompt() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream = lm
        .generate(&user_convo("   \n"), &GenParams::default(), &[])
        .await
        .expect("generate");
    let items = drain(&mut stream).await;
    assert!(items.is_empty(), "got {items:?}");
    mock.assert_quiet().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn dispatch_event_tail_renders_bracketed_and_assistant_tail_errors() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let convo = Conversation {
        messages: vec![
            Message::system("sys"),
            Message::user("start a task"),
            Message::Event {
                source: "dispatch".into(),
                kind: "completion".into(),
                content: "task pc-1 (agent research): done".into(),
            },
        ],
    };
    let mut stream = lm
        .generate(&convo, &GenParams::default(), &[])
        .await
        .expect("generate");
    let prompt = mock.next_request().await;
    assert_eq!(
        prompt.params["prompt"],
        "[dispatch/completion] task pc-1 (agent research): done"
    );
    mock.turn_complete("sess-test", "completed", "spoken follow-up");
    mock.respond_ok(prompt.id, json!({ "content": "spoken follow-up" }));
    let _ = drain(&mut stream).await;

    let bad = Conversation {
        messages: vec![Message::assistant("dangling")],
    };
    let Err(error) = lm.generate(&bad, &GenParams::default(), &[]).await else {
        panic!("assistant tail is a protocol violation");
    };
    assert!(matches!(error, LmError::Engine(_)));
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_sends_session_cancel_and_suppresses_stale_deltas() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream = lm
        .generate(
            &user_convo("tell me a long story"),
            &GenParams::default(),
            &[],
        )
        .await
        .expect("generate");
    let prompt = mock.next_request().await;

    mock.chunk("sess-test", "Once upon");
    assert_eq!(
        next_delta(&mut stream).await,
        Ok(ModelDelta::Text("Once upon".into()))
    );

    lm.cancel();
    let cancel = mock.next_request().await;
    assert_eq!(cancel.method, "session/cancel");
    assert_eq!(cancel.params["session_id"], "sess-test");

    // Anything the daemon streams after the cancel must not surface.
    mock.chunk("sess-test", " a time");
    mock.turn_complete("sess-test", "cancelled", "Once upon a time");
    mock.respond_ok(
        cancel.id,
        json!({ "session_id": "sess-test", "cancelled": true }),
    );
    mock.respond_ok(prompt.id, json!({ "content": "Once upon a time" }));

    let rest = drain(&mut stream).await;
    assert!(rest.is_empty(), "stale deltas leaked: {rest:?}");

    // A second cancel is a no-op: no further daemon traffic.
    lm.cancel();
    mock.assert_quiet().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn generate_during_cancelled_turn_waits_for_the_terminal() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");

    let mut stream1 = lm
        .generate(&user_convo("first"), &GenParams::default(), &[])
        .await
        .expect("generate 1");
    let prompt1 = mock.next_request().await;
    assert_eq!(prompt1.params["prompt"], "first");

    lm.cancel();
    let cancel = mock.next_request().await;
    assert_eq!(cancel.method, "session/cancel");

    // The replacement utterance arrives while turn 1 is still draining.
    let mut stream2 = lm
        .generate(&user_convo("second"), &GenParams::default(), &[])
        .await
        .expect("generate 2");

    // No second prompt may go out before turn 1's terminal.
    mock.assert_quiet().await;

    mock.turn_complete("sess-test", "cancelled", "");
    mock.respond_ok(prompt1.id, json!({ "content": "" }));
    mock.respond_ok(cancel.id, json!({ "cancelled": true }));

    let prompt2 = mock.next_request().await;
    assert_eq!(prompt2.method, "session/prompt");
    assert_eq!(prompt2.params["prompt"], "second");

    mock.chunk("sess-test", "second reply");
    mock.turn_complete("sess-test", "completed", "second reply");
    mock.respond_ok(prompt2.id, json!({ "content": "second reply" }));

    assert!(drain(&mut stream1).await.is_empty());
    assert_eq!(
        drain(&mut stream2).await,
        vec![Ok(ModelDelta::Text("second reply".into()))]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn reconnect_restores_identity_and_session() {
    let mut mock = MockDaemon::spawn();
    let (lm, _source) = connect(mock.config("voice")).await.expect("connect");
    let _ = mock.next_handshake().await; // initialize
    let _ = mock.next_handshake().await; // session/new

    // Kill the connection mid-turn: the stream fails recoverably.
    let mut stream = lm
        .generate(&user_convo("hi"), &GenParams::default(), &[])
        .await
        .expect("generate");
    let _prompt = mock.next_request().await;
    mock.drop_connection();
    let items = drain(&mut stream).await;
    assert_eq!(items.len(), 1);
    assert!(
        matches!(&items[0], Err(LmError::Engine(_))),
        "got {items:?}"
    );

    // The actor reconnects: initialize echoes the saved identity, and
    // session/new reattaches the same session id.
    let init = mock.next_handshake().await;
    assert_eq!(init.method, "initialize");
    assert_eq!(init.params["tui_id"], "tui-1");
    assert_eq!(init.params["tui_sig"], "sig-1");
    let renew = mock.next_handshake().await;
    assert_eq!(renew.method, "session/new");
    assert_eq!(renew.params["session_id"], "sess-test");

    // And the next turn works.
    let mut stream = lm
        .generate(&user_convo("again"), &GenParams::default(), &[])
        .await
        .expect("generate after reconnect");
    let prompt = mock.next_request().await;
    assert_eq!(prompt.params["prompt"], "again");
    mock.chunk("sess-test", "back");
    mock.turn_complete("sess-test", "completed", "back");
    mock.respond_ok(prompt.id, json!({ "content": "back" }));
    assert_eq!(
        drain(&mut stream).await,
        vec![Ok(ModelDelta::Text("back".into()))]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_the_handles_shuts_the_worker_down() {
    let mock = MockDaemon::spawn();
    let (lm, source) = connect(mock.config("voice")).await.expect("connect");
    // Drop joins the worker thread; the test completing is the assertion —
    // a hung shutdown would time the whole test binary out.
    drop(source);
    drop(lm);
}
