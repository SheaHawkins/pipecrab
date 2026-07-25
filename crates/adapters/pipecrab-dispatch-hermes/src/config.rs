//! [`HermesConfig`]: how to reach a Hermes Agent gateway and how hard to poll it.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use url::Url;

/// The default gateway address a local `hermes gateway` listens on.
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8642";
/// How often the poller sweeps active runs. 1s sits comfortably inside the
/// gateway's terminal-state retention window, so a completed run is observed
/// before its state expires.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Per-HTTP-request timeout. A poll tick issues one GET per active run
/// sequentially, so an unresponsive gateway degrades a tick toward
/// `active_runs * request_timeout` — kept modest for that reason.
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Connection parameters for a Hermes Agent gateway.
///
/// Construct with [`HermesConfig::new`] (base URL defaulted) and override the
/// timings with the builder methods, or fill the public fields directly.
///
/// The [`api_key`](Self::api_key) is the gateway's `API_SERVER_KEY` bearer
/// token. It is never logged: [`HermesConfig`]'s [`Debug`] redacts it, and it
/// appears in no error message.
#[derive(Clone)]
pub struct HermesConfig {
    /// Gateway base URL. Defaults to `http://127.0.0.1:8642`.
    pub base_url: Url,
    /// `API_SERVER_KEY` bearer token. Redacted from [`Debug`]; never logged.
    pub api_key: Arc<str>,
    /// How often to poll every active run. Defaults to 1s.
    pub poll_interval: Duration,
    /// Timeout for a single HTTP request. Defaults to 10s.
    pub request_timeout: Duration,
}

impl HermesConfig {
    /// A config for `api_key` against the default local gateway
    /// (`http://127.0.0.1:8642`) with default timings.
    pub fn new(api_key: impl Into<Arc<str>>) -> Self {
        Self {
            base_url: Url::parse(DEFAULT_BASE_URL).expect("the default base URL is valid"),
            api_key: api_key.into(),
            poll_interval: DEFAULT_POLL_INTERVAL,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }

    /// Point at a specific gateway URL.
    pub fn with_base_url(mut self, base_url: Url) -> Self {
        self.base_url = base_url;
        self
    }

    /// Override the poll interval.
    pub fn with_poll_interval(mut self, poll_interval: Duration) -> Self {
        self.poll_interval = poll_interval;
        self
    }

    /// Override the per-request timeout.
    pub fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = request_timeout;
        self
    }
}

impl fmt::Debug for HermesConfig {
    /// Redacts [`api_key`](Self::api_key); the token must never reach a log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HermesConfig")
            .field("base_url", &self.base_url.as_str())
            .field("api_key", &"<redacted>")
            .field("poll_interval", &self.poll_interval)
            .field("request_timeout", &self.request_timeout)
            .finish()
    }
}
