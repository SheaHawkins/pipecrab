//! The connection actor: one named thread running a private current-thread
//! tokio runtime that owns the daemon connection for the adapter's lifetime.
//!
//! Owning a private runtime keeps the pipeline free to be driven by any
//! executor, and gives the delegation poller a place to live. The actor
//! serializes turns (a barge-in followed by a fast next utterance cannot
//! interleave two prompts), tags every turn with the cancellation epoch so a
//! cancel racing in-flight notifications cannot leak text into the next
//! turn, and reconnects with capped backoff when the socket drops —
//! re-initializing with the saved identity and reattaching to the same
//! session id, so the conversation survives restarts on both sides.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures::StreamExt;
use futures::channel::{mpsc as fmpsc, oneshot};
use pipecrab_core::DispatchEvent;
use pipecrab_lm::{LmError, ModelDelta};
use tokio::sync::{mpsc, watch};
use tokio::time::Instant;

use crate::client::{Connection, Incoming};
use crate::config::{ZeroclawLmBuildError, ZeroclawLmConfig};
use crate::protocol::{
    InitializeParams, InitializeResult, SessionIdParams, SessionNewParams, SessionNewResult,
    SessionPromptParams, SessionPromptResult, SessionUpdate, TurnOutcome, method,
};
use crate::source::{PollCancel, run_poller};

/// Bound on buffered delegation events; workers await sends, so a stalled
/// pipeline back-pressures the poller instead of ballooning.
const EVENT_CHANNEL_CAPACITY: usize = 64;
/// Ceiling on each handshake request during bootstrap and reconnect.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// How long after an OK `session/prompt` response to keep waiting for the
/// terminal `turn_complete` before treating the response as the terminal.
/// The daemon writes the terminal notification first on the same ordered
/// socket, so this fires only if that ordering ever changes.
const RESPONSE_GRACE: Duration = Duration::from_secs(5);
/// Reconnect backoff bounds.
const RECONNECT_MIN: Duration = Duration::from_millis(250);
const RECONNECT_MAX: Duration = Duration::from_secs(5);

/// A command from the handle to the actor.
pub(crate) enum Command {
    Generate(GenerateCmd),
    Shutdown,
}

/// One requested generation.
pub(crate) struct GenerateCmd {
    pub(crate) prompt: String,
    pub(crate) epoch: u64,
    pub(crate) deltas: fmpsc::UnboundedSender<Result<ModelDelta, LmError>>,
}

/// What the session bootstrap established.
#[derive(Debug, Clone)]
pub(crate) struct Bootstrap {
    pub(crate) session_id: String,
    pub(crate) workspace_dir: PathBuf,
}

/// Everything the worker hands back through the readiness channel.
pub(crate) struct WorkerHandles {
    pub(crate) bootstrap: Bootstrap,
    pub(crate) events: mpsc::Receiver<DispatchEvent>,
    pub(crate) poll_cancel: Arc<PollCancel>,
}

/// Spawn the worker thread. The readiness oneshot resolves once the socket
/// is dialed, `initialize` and `session/new` have succeeded, and the
/// delegation poller is running — or with the error that prevented it.
pub(crate) fn spawn_worker(
    config: ZeroclawLmConfig,
    cmd_rx: fmpsc::UnboundedReceiver<Command>,
    cancel_rx: watch::Receiver<u64>,
    epoch: Arc<AtomicU64>,
    ready_tx: oneshot::Sender<Result<WorkerHandles, ZeroclawLmBuildError>>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("pipecrab-lm-zeroclaw".into())
        .spawn(move || worker_main(config, cmd_rx, cancel_rx, epoch, ready_tx))
        .expect("spawning the zeroclaw worker thread failed")
}

fn worker_main(
    config: ZeroclawLmConfig,
    cmd_rx: fmpsc::UnboundedReceiver<Command>,
    cancel_rx: watch::Receiver<u64>,
    epoch: Arc<AtomicU64>,
    ready_tx: oneshot::Sender<Result<WorkerHandles, ZeroclawLmBuildError>>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = ready_tx.send(Err(ZeroclawLmBuildError::Handshake(format!(
                "building the worker runtime failed: {error}"
            ))));
            return;
        }
    };

    runtime.block_on(async move {
        let socket = config.resolve_socket_path();
        let mut identity = Identity::default();
        let desired_session = config
            .session_id
            .as_deref()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("pc-voice-{}", uuid::Uuid::new_v4().simple()));

        let (conn, bootstrap) =
            match establish(&socket, &config, &mut identity, &desired_session).await {
                Ok(established) => established,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };

        let (events_tx, events_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let turn_settled = Arc::new(tokio::sync::Notify::new());
        let poll_cancel = Arc::new(PollCancel::default());
        tokio::spawn(run_poller(
            bootstrap.workspace_dir.join("delegate_results"),
            config.poll.clone(),
            chrono::Utc::now(),
            Arc::clone(&turn_settled),
            Arc::clone(&poll_cancel),
            events_tx,
        ));

        let session_id = bootstrap.session_id.clone();
        if ready_tx
            .send(Ok(WorkerHandles {
                bootstrap,
                events: events_rx,
                poll_cancel,
            }))
            .is_err()
        {
            return; // connect() caller vanished before readiness
        }

        Actor {
            config,
            socket,
            identity,
            session_id,
            epoch,
            cancel_rx,
            turn_settled,
            conn: Some(conn),
            backoff: RECONNECT_MIN,
            next_attempt: Instant::now(),
        }
        .run(cmd_rx)
        .await;
    });
}

/// The daemon-assigned client identity, echoed on reconnect.
#[derive(Debug, Default, Clone)]
struct Identity {
    tui_id: Option<String>,
    tui_sig: Option<String>,
}

/// Dial, `initialize`, `session/new`. Used for both bootstrap and reconnect.
async fn establish(
    socket: &std::path::Path,
    config: &ZeroclawLmConfig,
    identity: &mut Identity,
    session_id: &str,
) -> Result<(Connection, Bootstrap), ZeroclawLmBuildError> {
    let mut conn = Connection::dial(socket)
        .await
        .map_err(|error| ZeroclawLmBuildError::Dial(format!("{}: {error}", socket.display())))?;

    let init = request(
        &mut conn,
        method::INITIALIZE,
        InitializeParams {
            protocol_version: 1,
            tui_id: identity.tui_id.clone(),
            tui_sig: identity.tui_sig.clone(),
        },
    )
    .await
    .map_err(ZeroclawLmBuildError::Handshake)?;
    let init: InitializeResult = serde_json::from_value(init).map_err(|error| {
        ZeroclawLmBuildError::Handshake(format!("bad initialize result: {error}"))
    })?;
    if init.tui_id.is_some() {
        identity.tui_id = init.tui_id;
    }
    if init.tui_sig.is_some() {
        identity.tui_sig = init.tui_sig;
    }

    let session = request(
        &mut conn,
        method::SESSION_NEW,
        SessionNewParams {
            agent_alias: config.agent_alias.to_string(),
            session_id: Some(session_id.to_owned()),
            exclude_memory: config.exclude_memory.then_some(true),
            chat_mode: "chat",
        },
    )
    .await
    .map_err(ZeroclawLmBuildError::Session)?;
    let session: SessionNewResult = serde_json::from_value(session).map_err(|error| {
        ZeroclawLmBuildError::Session(format!("bad session/new result: {error}"))
    })?;
    if session.message_count > 0 {
        eprintln!(
            "pipecrab-lm-zeroclaw: reattached to session {} ({} message(s) of history)",
            session.session_id, session.message_count,
        );
    }

    Ok((
        conn,
        Bootstrap {
            session_id: session.session_id,
            workspace_dir: PathBuf::from(session.workspace_dir),
        },
    ))
}

/// One handshake request with a timeout: send, await the matching response.
async fn request(
    conn: &mut Connection,
    method: &str,
    params: impl serde::Serialize,
) -> Result<serde_json::Value, String> {
    let id = conn
        .send_request(method, params)
        .await
        .map_err(|error| format!("sending {method} failed: {error}"))?;
    tokio::time::timeout(HANDSHAKE_TIMEOUT, conn.await_response(id))
        .await
        .map_err(|_| format!("{method} timed out after {HANDSHAKE_TIMEOUT:?}"))?
        .map_err(|error| format!("{method} failed: {error}"))
}

/// How one turn ended, from the actor's perspective.
enum TurnEnd {
    /// Terminal event observed (or the turn was rejected/suppressed).
    Done,
    /// The connection died mid-turn; the actor should go into reconnect.
    ConnDown,
    /// Shutdown was requested mid-turn; the actor should exit.
    Shutdown,
}

struct Actor {
    config: ZeroclawLmConfig,
    socket: PathBuf,
    identity: Identity,
    session_id: String,
    epoch: Arc<AtomicU64>,
    cancel_rx: watch::Receiver<u64>,
    turn_settled: Arc<tokio::sync::Notify>,
    conn: Option<Connection>,
    backoff: Duration,
    next_attempt: Instant,
}

impl Actor {
    async fn run(mut self, mut cmd_rx: fmpsc::UnboundedReceiver<Command>) {
        loop {
            if self.conn.is_some() {
                let up = self.tick_connected(&mut cmd_rx).await;
                if !up {
                    break;
                }
            } else {
                let up = self.tick_reconnecting(&mut cmd_rx).await;
                if !up {
                    break;
                }
            }
        }
    }

    /// One event while connected. Returns `false` to exit the actor.
    async fn tick_connected(&mut self, cmd_rx: &mut fmpsc::UnboundedReceiver<Command>) -> bool {
        let conn = self
            .conn
            .as_mut()
            .expect("tick_connected requires a connection");
        tokio::select! {
            biased;
            command = cmd_rx.next() => match command {
                None | Some(Command::Shutdown) => false,
                Some(Command::Generate(generate)) => {
                    // A barge-in followed by a fast next utterance queues one
                    // generation behind the cancelled turn's terminal; run
                    // queued turns back to back.
                    let mut pending = Some(generate);
                    while let Some(generate) = pending.take() {
                        let (end, queued) = self.turn(cmd_rx, generate).await;
                        self.turn_settled.notify_waiters();
                        match end {
                            TurnEnd::Done => pending = queued,
                            TurnEnd::ConnDown => {
                                if let Some(queued) = queued {
                                    let _ = queued.deltas.unbounded_send(Err(LmError::Engine(
                                        "zeroclaw daemon unavailable; reconnecting".into(),
                                    )));
                                }
                                self.mark_down();
                                break;
                            }
                            TurnEnd::Shutdown => return false,
                        }
                    }
                    true
                }
            },
            incoming = conn.incoming.recv() => {
                match incoming {
                    // Between turns: trailing prompt/cancel responses and
                    // stale updates are expected noise.
                    Some(Incoming::Response { .. }) | Some(Incoming::Update(_)) => {}
                    Some(Incoming::Closed(reason)) => {
                        eprintln!("pipecrab-lm-zeroclaw: {reason}; reconnecting");
                        self.mark_down();
                    }
                    None => self.mark_down(),
                }
                true
            },
            _ = self.cancel_rx.changed() => {
                // Idle cancel: nothing in flight; mark the change seen.
                let _ = *self.cancel_rx.borrow_and_update();
                true
            },
        }
    }

    /// One event while disconnected. Returns `false` to exit the actor.
    async fn tick_reconnecting(&mut self, cmd_rx: &mut fmpsc::UnboundedReceiver<Command>) -> bool {
        tokio::select! {
            biased;
            command = cmd_rx.next() => match command {
                None | Some(Command::Shutdown) => false,
                Some(Command::Generate(generate)) => {
                    // Fail fast and recoverably; the next utterance retries.
                    let _ = generate.deltas.unbounded_send(Err(LmError::Engine(
                        "zeroclaw daemon unavailable; reconnecting".into(),
                    )));
                    self.turn_settled.notify_waiters();
                    true
                }
            },
            _ = tokio::time::sleep_until(self.next_attempt) => {
                match establish(&self.socket, &self.config, &mut self.identity, &self.session_id).await {
                    Ok((conn, bootstrap)) => {
                        if bootstrap.session_id != self.session_id {
                            eprintln!(
                                "pipecrab-lm-zeroclaw: daemon reattached session {} as {}",
                                self.session_id, bootstrap.session_id,
                            );
                            self.session_id = bootstrap.session_id;
                        }
                        self.conn = Some(conn);
                        self.backoff = RECONNECT_MIN;
                    }
                    Err(error) => {
                        eprintln!("pipecrab-lm-zeroclaw: reconnect failed: {error}");
                        self.backoff = (self.backoff * 2).min(RECONNECT_MAX);
                        self.next_attempt = Instant::now() + self.backoff;
                    }
                }
                true
            },
        }
    }

    fn mark_down(&mut self) {
        self.conn = None;
        self.backoff = RECONNECT_MIN;
        self.next_attempt = Instant::now();
    }

    /// Drive one `session/prompt` turn to its terminal. Returns the turn's
    /// end plus at most one generation that arrived (and was queued) while
    /// this turn was still draining.
    async fn turn(
        &mut self,
        cmd_rx: &mut fmpsc::UnboundedReceiver<Command>,
        generate: GenerateCmd,
    ) -> (TurnEnd, Option<GenerateCmd>) {
        let GenerateCmd {
            prompt,
            epoch: turn_epoch,
            deltas,
        } = generate;
        // A cancel between the stage's epoch snapshot and here supersedes the
        // turn before it starts: the stream just ends empty.
        if self.epoch.load(Ordering::Acquire) != turn_epoch {
            return (TurnEnd::Done, None);
        }
        // Any pending watch change at this point implies the epoch mismatch
        // above, so marking it seen cannot swallow a live cancel.
        let _ = *self.cancel_rx.borrow_and_update();

        let conn = self.conn.as_mut().expect("turn requires a live connection");
        let prompt_id = match conn
            .send_request(
                method::SESSION_PROMPT,
                SessionPromptParams {
                    session_id: self.session_id.clone(),
                    prompt,
                },
            )
            .await
        {
            Ok(id) => id,
            Err(error) => {
                let _ = deltas.unbounded_send(Err(LmError::Engine(format!(
                    "sending session/prompt failed: {error}"
                ))));
                return (TurnEnd::ConnDown, None);
            }
        };

        let mut chunks_seen = false;
        let mut suppressed = false;
        let mut cancel_requested = false;
        let mut own_request_ids: HashSet<u64> = HashSet::from([prompt_id]);
        let mut fallback_content: Option<String> = None;
        let mut queued: Option<GenerateCmd> = None;
        // Whether the text emitted so far ends in whitespace. The daemon
        // concatenates a multi-iteration turn's texts with no separator, so
        // "…right now." + "I've started…" reads as one unsplittable blob to
        // the sentence chunker and nothing reaches TTS until the turn's
        // final flush. A tool boundary injects the missing whitespace —
        // emitted at the boundary itself, so the completed sentence starts
        // synthesizing while the tool is still running.
        let mut text_ends_in_whitespace = true;

        loop {
            tokio::select! {
                biased;
                command = cmd_rx.next() => match command {
                    // Shutdown mid-turn: abandon the turn; the daemon
                    // finishes it alone and the trailing traffic is ignored
                    // by whoever connects next.
                    None | Some(Command::Shutdown) => return (TurnEnd::Shutdown, None),
                    Some(Command::Generate(next)) => {
                        // A barge-in's replacement utterance arrives while
                        // the cancelled turn is still draining: queue exactly
                        // one; a second concurrent generation is a caller
                        // bug, surfaced recoverably.
                        if queued.is_none() {
                            queued = Some(next);
                        } else {
                            let _ = next.deltas.unbounded_send(Err(LmError::Engine(
                                "a zeroclaw turn is already in flight and one is queued".into(),
                            )));
                        }
                    }
                },
                _ = self.cancel_rx.changed() => {
                    let latest = *self.cancel_rx.borrow_and_update();
                    if latest != turn_epoch && !cancel_requested {
                        cancel_requested = true;
                        suppressed = true;
                        match conn
                            .send_request(
                                method::SESSION_CANCEL,
                                SessionIdParams { session_id: self.session_id.clone() },
                            )
                            .await
                        {
                            Ok(id) => { own_request_ids.insert(id); }
                            Err(_) => return (TurnEnd::ConnDown, queued),
                        }
                    }
                },
                incoming = conn.incoming.recv() => match incoming {
                    None => {
                        if !suppressed {
                            let _ = deltas.unbounded_send(Err(LmError::Engine(
                                "connection reader ended mid-turn".into(),
                            )));
                        }
                        return (TurnEnd::ConnDown, queued);
                    }
                    Some(Incoming::Closed(reason)) => {
                        if !suppressed {
                            let _ = deltas.unbounded_send(Err(LmError::Engine(format!(
                                "connection lost mid-turn: {reason}"
                            ))));
                        }
                        return (TurnEnd::ConnDown, queued);
                    }
                    Some(Incoming::Response { id, result }) => {
                        if id == prompt_id {
                            match result {
                                // An early error response (blank prompt,
                                // session gone) is the turn's terminal.
                                Err(error) => {
                                    if !suppressed {
                                        let _ = deltas.unbounded_send(Err(LmError::Engine(
                                            format!("session/prompt rejected: {error}"),
                                        )));
                                    }
                                    return (TurnEnd::Done, queued);
                                }
                                // The daemon writes turn_complete before this
                                // response; tolerate the reverse via the
                                // grace timer below.
                                Ok(value) => {
                                    let parsed: SessionPromptResult =
                                        serde_json::from_value(value).unwrap_or(
                                            SessionPromptResult { content: String::new() },
                                        );
                                    fallback_content = Some(parsed.content);
                                }
                            }
                        } else if !own_request_ids.remove(&id) {
                            // A response to a request from a previous life of
                            // this connection; ignore.
                        }
                    }
                    Some(Incoming::Update(update)) => {
                        if update.session_id() != Some(self.session_id.as_str()) {
                            continue;
                        }
                        match update {
                            SessionUpdate::MessageChunk { text, .. } => {
                                if text.is_empty() {
                                    continue;
                                }
                                chunks_seen = true;
                                text_ends_in_whitespace =
                                    text.chars().last().is_some_and(char::is_whitespace);
                                if !suppressed
                                    && deltas
                                        .unbounded_send(Ok(ModelDelta::Text(text.into())))
                                        .is_err()
                                {
                                    // The stage dropped the stream (barge-in
                                    // teardown); stop forwarding, keep
                                    // draining to the terminal.
                                    suppressed = true;
                                }
                            }
                            SessionUpdate::ToolCall { tool_call_id, name, raw_input, .. } => {
                                if !suppressed
                                    && chunks_seen
                                    && !text_ends_in_whitespace
                                {
                                    // The speakable gap (see
                                    // `text_ends_in_whitespace` above).
                                    let _ = deltas
                                        .unbounded_send(Ok(ModelDelta::Text(" ".into())));
                                    text_ends_in_whitespace = true;
                                }
                                if !suppressed && self.config.surface_tool_calls {
                                    let id = if tool_call_id.is_empty() {
                                        format!("zc-{}", uuid::Uuid::new_v4().simple())
                                    } else {
                                        tool_call_id
                                    };
                                    // Non-object arguments (or an empty name)
                                    // are not surfaceable; skip quietly —
                                    // execution is the daemon's job either way.
                                    if let Ok(delta) = ModelDelta::tool_call(id, name, raw_input) {
                                        let _ = deltas.unbounded_send(Ok(delta));
                                    }
                                }
                            }
                            SessionUpdate::ToolResult { .. }
                                if !suppressed && chunks_seen && !text_ends_in_whitespace =>
                            {
                                // Some providers surface only the result side
                                // of a tool round; the gap is owed either way.
                                let _ =
                                    deltas.unbounded_send(Ok(ModelDelta::Text(" ".into())));
                                text_ends_in_whitespace = true;
                            }
                            SessionUpdate::TurnComplete { outcome, content, .. } => {
                                match outcome {
                                    TurnOutcome::Completed => {
                                        if !suppressed
                                            && !chunks_seen
                                            && !content.trim().is_empty()
                                        {
                                            eprintln!(
                                                "pipecrab-lm-zeroclaw: turn was not streamed by \
                                                 the provider; time-to-first-audio degrades \
                                                 (check the agent profile's provider streaming \
                                                 support)"
                                            );
                                            let _ = deltas
                                                .unbounded_send(Ok(ModelDelta::Text(content.into())));
                                        }
                                    }
                                    TurnOutcome::Cancelled => {}
                                    TurnOutcome::Failed => {
                                        if !suppressed {
                                            let message = if content.trim().is_empty() {
                                                "zeroclaw turn failed".to_owned()
                                            } else {
                                                content
                                            };
                                            let _ = deltas
                                                .unbounded_send(Err(LmError::Engine(message)));
                                        }
                                    }
                                }
                                return (TurnEnd::Done, queued);
                            }
                            SessionUpdate::ApprovalRequest { tool_name, .. } => {
                                eprintln!(
                                    "pipecrab-lm-zeroclaw: approval_request for tool \
                                     {tool_name:?} cannot be answered from the voice loop; the \
                                     turn stalls until it times out (remove approval gating \
                                     from the voice agent profile)"
                                );
                            }
                            SessionUpdate::HistoryTrimmed { dropped_messages, reason, .. } => {
                                eprintln!(
                                    "pipecrab-lm-zeroclaw: daemon trimmed {dropped_messages} \
                                     message(s) from session history ({reason})"
                                );
                            }
                            SessionUpdate::ThoughtChunk { .. }
                            | SessionUpdate::ToolResult { .. }
                            | SessionUpdate::ContextUsage { .. }
                            | SessionUpdate::Plan { .. }
                            | SessionUpdate::Unknown { .. } => {}
                        }
                    }
                },
                _ = tokio::time::sleep(RESPONSE_GRACE), if fallback_content.is_some() => {
                    let content = fallback_content.take().unwrap_or_default();
                    if !suppressed && !chunks_seen && !content.trim().is_empty() {
                        eprintln!(
                            "pipecrab-lm-zeroclaw: turn terminal arrived as the RPC response \
                             only (no turn_complete); treating it as completed"
                        );
                        let _ = deltas.unbounded_send(Ok(ModelDelta::Text(content.into())));
                    }
                    return (TurnEnd::Done, queued);
                },
            }
        }
    }
}
