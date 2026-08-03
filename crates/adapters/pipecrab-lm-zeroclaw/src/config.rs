//! [`ZeroclawLmConfig`] and [`PollConfig`]: everything the adapter needs to
//! reach a daemon and pace the delegation poller. The *agent profile*
//! (provider streaming, tool registry, delegation policy) lives in ZeroClaw's
//! own configuration — see the crate docs for what a voice profile requires.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// How the delegation-results poller paces itself.
#[derive(Debug, Clone)]
pub struct PollConfig {
    /// Scan cadence while background tasks are pending.
    pub interval: Duration,
    /// Relaxed cadence once a task set has been pending for a while
    /// (30 seconds), so a long task does not keep the tight cadence alive.
    pub settle_backoff: Duration,
    /// A task still non-terminal this long after first seen is reported as a
    /// failure and dropped — there is no reaper for the result files.
    pub stale_after: Duration,
}

impl Default for PollConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(500),
            settle_backoff: Duration::from_secs(2),
            stale_after: Duration::from_secs(15 * 60),
        }
    }
}

/// Configuration for [`connect`](crate::connect).
#[derive(Debug, Clone)]
pub struct ZeroclawLmConfig {
    /// The daemon's local RPC socket. `None` resolves `$ZEROCLAW_SOCKET`,
    /// then `$ZEROCLAW_CONFIG_DIR/data/daemon.sock`, then the daemon's
    /// default `~/.zeroclaw/data/daemon.sock`.
    pub socket_path: Option<PathBuf>,
    /// The ZeroClaw agent alias the session runs as.
    pub agent_alias: Arc<str>,
    /// Stable session id, so the conversation survives restarts on both
    /// sides and the TUI finds one session, not many. `None` mints
    /// `pc-voice-{uuid}` fresh each run.
    pub session_id: Option<Arc<str>>,
    /// Ask the daemon to skip semantic-memory injection for this session.
    pub exclude_memory: bool,
    /// Emit the daemon's tool calls as [`ModelDelta::ToolCall`]s for
    /// downstream observability ("Let me check that…" filler audio). They
    /// are never dispatched — tool execution is internal to the daemon.
    ///
    /// [`ModelDelta::ToolCall`]: pipecrab_lm::ModelDelta::ToolCall
    pub surface_tool_calls: bool,
    /// Delegation poller pacing.
    pub poll: PollConfig,
}

impl ZeroclawLmConfig {
    /// A configuration for `agent_alias` with every default: socket from the
    /// environment or `~/.zeroclaw/daemon.sock`, a freshly minted session id,
    /// memory injection on, tool calls surfaced.
    pub fn new(agent_alias: impl Into<Arc<str>>) -> Self {
        Self {
            socket_path: None,
            agent_alias: agent_alias.into(),
            session_id: None,
            exclude_memory: false,
            surface_tool_calls: true,
            poll: PollConfig::default(),
        }
    }

    /// Set an explicit daemon socket path.
    #[must_use]
    pub fn with_socket_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.socket_path = Some(path.into());
        self
    }

    /// Set a stable session id to reattach to across runs.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<Arc<str>>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Skip semantic-memory injection for this session.
    #[must_use]
    pub fn with_exclude_memory(mut self, exclude: bool) -> Self {
        self.exclude_memory = exclude;
        self
    }

    /// Surface or suppress observability tool-call deltas.
    #[must_use]
    pub fn with_surface_tool_calls(mut self, surface: bool) -> Self {
        self.surface_tool_calls = surface;
        self
    }

    /// Set the delegation poller pacing.
    #[must_use]
    pub fn with_poll(mut self, poll: PollConfig) -> Self {
        self.poll = poll;
        self
    }

    /// Resolve the effective socket path: explicit, `$ZEROCLAW_SOCKET`, or
    /// `~/.zeroclaw/daemon.sock`.
    pub(crate) fn resolve_socket_path(&self) -> PathBuf {
        if let Some(path) = &self.socket_path {
            return path.clone();
        }
        if let Ok(path) = std::env::var("ZEROCLAW_SOCKET")
            && !path.trim().is_empty()
        {
            return PathBuf::from(path);
        }
        // The daemon binds `{data_dir}/daemon.sock`, and its data dir is
        // `{config_dir}/data` — config dir from $ZEROCLAW_CONFIG_DIR, else
        // `~/.zeroclaw`.
        let config_dir = match std::env::var("ZEROCLAW_CONFIG_DIR") {
            Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir),
            _ => std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_default()
                .join(".zeroclaw"),
        };
        config_dir.join("data").join("daemon.sock")
    }
}

/// Why [`connect`](crate::connect) failed.
#[derive(Debug, Clone)]
pub enum ZeroclawLmBuildError {
    /// Dialing the daemon socket failed — is `zeroclaw daemon` running?
    Dial(String),
    /// The `initialize` handshake failed or timed out.
    Handshake(String),
    /// `session/new` was rejected — an unknown agent alias, most likely.
    Session(String),
}

impl std::fmt::Display for ZeroclawLmBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZeroclawLmBuildError::Dial(msg) => {
                write!(f, "dialing the zeroclaw daemon socket failed: {msg}")
            }
            ZeroclawLmBuildError::Handshake(msg) => {
                write!(f, "zeroclaw daemon handshake failed: {msg}")
            }
            ZeroclawLmBuildError::Session(msg) => {
                write!(f, "zeroclaw session bootstrap failed: {msg}")
            }
        }
    }
}

impl std::error::Error for ZeroclawLmBuildError {}
