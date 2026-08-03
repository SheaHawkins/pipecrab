//! ZeroClaw daemon [`LanguageModel`] adapter: a ZeroClaw agent stands in as
//! the pipeline's LM stage, and the voice pipeline becomes one more native
//! interaction mechanism in ZeroClaw.
//!
//! [`ZeroclawLm`] is a JSON-RPC peer of a running `zeroclaw daemon` — not an
//! embedded agent. The conversation is a first-class daemon session: it
//! survives restarts on both sides, appears in `session/list`, and the
//! ZeroClaw TUI (another peer of the same daemon) reads its transcript at
//! turn granularity while voice is running. This crate carries **no ZeroClaw
//! crate dependency**; the protocol subset is mirrored in [`mod@protocol`]'s
//! terms and drift is caught by the ignored integration test.
//!
//! Tool calling is internalized: the daemon's agent loop executes tools
//! inline during a turn, and asynchronous work goes through ZeroClaw's
//! `delegate` tool with `background: true`. No dispatch command ever leaves
//! the pipeline — `DispatchEgress` and `DispatchSink` are not used. The
//! [`ZeroclawDelegateSource`] is the re-entry path: it watches the session
//! workspace's `delegate_results/` and emits a
//! [`Completion`](pipecrab_core::DispatchEvent::Completion) or
//! [`Failure`](pipecrab_core::DispatchEvent::Failure) through
//! `DispatchIngress`, whose projection triggers the spoken follow-up.
//!
//! ```text
//! … → UserTurnGate → DispatchIngress<ZeroclawDelegateSource>
//!   → LmStage<ZeroclawLm>   (constructed WITHOUT tools)
//!   → SentenceChunker → Tts → …
//! ```
//!
//! # The agent profile is part of the contract
//!
//! Streaming granularity is inherited from the session's provider: the
//! daemon streams chunk events only when the active provider supports
//! streaming (and, with tools registered, streaming tool events). A
//! non-qualifying profile still works but delivers the whole reply as one
//! terminal delta — logged, never silent. The voice profile should also keep
//! the tool registry trimmed (inline tools stall speech), exclude
//! `spawn_subagent`, allow `delegate` in background mode, avoid
//! approval-gated tools, and explain bracketed `[dispatch/completion]`
//! messages in its system prompt.
//!
//! # Accepted divergences
//!
//! The daemon session's history is the source of truth. `LmStage`'s own
//! conversation is read only at its tail (see [`LanguageModel::generate`])
//! and grows as bounded dead weight. On barge-in the daemon keeps the
//! partial assistant text — the agent remembering what the user actually
//! heard beats pipecrab's no-commit convention for voice.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod client;
mod config;
mod protocol;
mod render;
mod source;
mod worker;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use futures::channel::{mpsc as fmpsc, oneshot};
use pipecrab_lm::{Conversation, GenParams, LanguageModel, LmError, ModelStream, ToolDefinition};
use tokio::sync::watch;

pub use config::{PollConfig, ZeroclawLmBuildError, ZeroclawLmConfig};
pub use source::ZeroclawDelegateSource;

use render::Rendered;
use worker::Command;

/// Connect to a running ZeroClaw daemon: dial the socket, `initialize`,
/// bootstrap the session with `session/new`, start the delegation poller,
/// and return the [`LanguageModel`] handle and the
/// [`DispatchSource`](pipecrab_dispatch::DispatchSource) wired to one
/// workspace and one turn-settled notifier.
///
/// The returned halves are independent to drop; the worker thread lives as
/// long as the last [`ZeroclawLm`] clone.
pub async fn connect(
    config: ZeroclawLmConfig,
) -> Result<(ZeroclawLm, ZeroclawDelegateSource), ZeroclawLmBuildError> {
    let (cmd_tx, cmd_rx) = fmpsc::unbounded();
    let (cancel_tx, cancel_rx) = watch::channel(0u64);
    let epoch = Arc::new(AtomicU64::new(0));
    let (ready_tx, ready_rx) = oneshot::channel();

    let thread = worker::spawn_worker(config, cmd_rx, cancel_rx, Arc::clone(&epoch), ready_tx);

    let handles = match ready_rx.await {
        Ok(Ok(handles)) => handles,
        Ok(Err(error)) => {
            let _ = thread.join();
            return Err(error);
        }
        Err(_canceled) => {
            let _ = thread.join();
            return Err(ZeroclawLmBuildError::Handshake(
                "the worker exited before completing the handshake".into(),
            ));
        }
    };

    let lm = ZeroclawLm {
        shared: Arc::new(Shared {
            cmd_tx,
            cancel_tx,
            epoch,
            session_id: handles.bootstrap.session_id.clone(),
            workspace_dir: handles.bootstrap.workspace_dir.clone(),
            warned_params: AtomicBool::new(false),
            thread: Mutex::new(Some(thread)),
        }),
    };
    let source = ZeroclawDelegateSource::new(handles.events, handles.poll_cancel);
    Ok((lm, source))
}

/// Shared state behind every [`ZeroclawLm`] clone. Dropping the last clone
/// shuts the actor down and joins the worker thread.
struct Shared {
    cmd_tx: fmpsc::UnboundedSender<Command>,
    cancel_tx: watch::Sender<u64>,
    epoch: Arc<AtomicU64>,
    session_id: String,
    workspace_dir: PathBuf,
    warned_params: AtomicBool,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Drop for Shared {
    fn drop(&mut self) {
        // Wake an in-flight turn, ask the actor to exit, then join.
        let next = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        let _ = self.cancel_tx.send(next);
        let _ = self.cmd_tx.unbounded_send(Command::Shutdown);
        if let Ok(mut slot) = self.thread.lock()
            && let Some(handle) = slot.take()
        {
            let _ = handle.join();
        }
    }
}

/// A [`LanguageModel`] over one ZeroClaw daemon session.
///
/// A thin cloneable handle: the connection actor owns the socket on its own
/// runtime, [`generate`](LanguageModel::generate) renders the conversation
/// tail into a `session/prompt` and returns the mapped `session/update`
/// stream, and [`cancel`](LanguageModel::cancel) is the synchronous barge-in
/// path (`session/cancel`, epoch-tagged delta suppression).
#[derive(Clone)]
pub struct ZeroclawLm {
    shared: Arc<Shared>,
}

impl ZeroclawLm {
    /// The daemon session this handle drives — the id the TUI sees in
    /// `session/list`.
    pub fn session_id(&self) -> &str {
        &self.shared.session_id
    }

    /// The session's workspace directory, as reported by `session/new`.
    /// [`connect`] already points the delegation poller at
    /// `{workspace_dir}/delegate_results`.
    pub fn workspace_dir(&self) -> &Path {
        &self.shared.workspace_dir
    }
}

#[async_trait]
impl LanguageModel for ZeroclawLm {
    async fn generate(
        &self,
        conversation: &Conversation,
        params: &GenParams,
        tools: &[ToolDefinition],
    ) -> Result<ModelStream, LmError> {
        if !tools.is_empty() {
            return Err(LmError::Engine(
                "tool definitions are managed by the ZeroClaw agent profile; \
                 construct LmStage without tools"
                    .into(),
            ));
        }
        if *params != GenParams::default()
            && !self.shared.warned_params.swap(true, Ordering::AcqRel)
        {
            eprintln!(
                "pipecrab-lm-zeroclaw: GenParams are ignored; sampling and budgets are \
                 governed by the ZeroClaw agent profile"
            );
        }

        match render::render_last(conversation)? {
            Rendered::Empty => Ok(futures::stream::empty().boxed()),
            Rendered::Prompt(prompt) => {
                let epoch = self.shared.epoch.load(Ordering::Acquire);
                let (deltas_tx, deltas_rx) = fmpsc::unbounded();
                self.shared
                    .cmd_tx
                    .unbounded_send(Command::Generate(worker::GenerateCmd {
                        prompt,
                        epoch,
                        deltas: deltas_tx,
                    }))
                    .map_err(|_| {
                        LmError::Engine("the zeroclaw worker is no longer running".into())
                    })?;
                Ok(deltas_rx.boxed())
            }
        }
    }

    fn cancel(&self) {
        let next = self.shared.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        // Fails only when the actor is already gone; nothing to cancel then.
        let _ = self.shared.cancel_tx.send(next);
    }

    async fn save_state(&self) -> Result<Vec<u8>, LmError> {
        // Session durability is the daemon's job; there is no client-side
        // state worth checkpointing.
        Ok(Vec::new())
    }

    async fn load_state(&self, _blob: &[u8]) -> Result<(), LmError> {
        Ok(())
    }
}
