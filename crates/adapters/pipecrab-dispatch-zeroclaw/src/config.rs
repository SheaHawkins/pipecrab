//! [`ZeroclawConfig`]: how to reach a ZeroClaw gateway and how long to wait on
//! its synchronous webhook.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use url::Url;

/// The default gateway address a local `zeroclaw gateway` listens on.
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:42617";
/// Per-request timeout. One `POST /webhook` spans a *whole agent turn* — the
/// gateway answers only once the turn finishes — so the default is generous
/// rather than snappy.
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Connection parameters for a ZeroClaw gateway.
///
/// Construct with [`ZeroclawConfig::new`] (base URL defaulted, no auth) and
/// layer on what the gateway requires, or fill the public fields directly.
///
/// Both credentials are optional because each is a gateway-side choice: a
/// gateway run without pairing needs no [`token`](Self::token), and
/// [`webhook_secret`](Self::webhook_secret) is an additional layer only some
/// deployments enable. Neither is ever logged: [`ZeroclawConfig`]'s [`Debug`]
/// redacts them, and they appear in no error message.
#[derive(Clone)]
pub struct ZeroclawConfig {
    /// Gateway base URL. Defaults to `http://127.0.0.1:42617`.
    pub base_url: Url,
    /// Pairing bearer token (from `POST /pair`), if the gateway requires
    /// pairing. Redacted from [`Debug`]; never logged.
    pub token: Option<Arc<str>>,
    /// The gateway's `X-Webhook-Secret` value, if configured. Redacted from
    /// [`Debug`]; never logged.
    pub webhook_secret: Option<Arc<str>>,
    /// A configured agent alias to dispatch to (`?agent=<alias>`), or `None`
    /// for the gateway's default agent.
    pub agent: Option<Arc<str>>,
    /// Timeout for a single HTTP request — one whole agent turn. Defaults to
    /// 300s.
    pub request_timeout: Duration,
}

impl ZeroclawConfig {
    /// A config for the default local gateway (`http://127.0.0.1:42617`) with
    /// no credentials and the default timeout.
    pub fn new() -> Self {
        Self {
            base_url: Url::parse(DEFAULT_BASE_URL).expect("the default base URL is valid"),
            token: None,
            webhook_secret: None,
            agent: None,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }

    /// Point at a specific gateway URL.
    pub fn with_base_url(mut self, base_url: Url) -> Self {
        self.base_url = base_url;
        self
    }

    /// Authenticate with a pairing bearer token.
    pub fn with_token(mut self, token: impl Into<Arc<str>>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Send an `X-Webhook-Secret` header on every request.
    pub fn with_webhook_secret(mut self, secret: impl Into<Arc<str>>) -> Self {
        self.webhook_secret = Some(secret.into());
        self
    }

    /// Dispatch to a specific configured agent alias.
    pub fn with_agent(mut self, agent: impl Into<Arc<str>>) -> Self {
        self.agent = Some(agent.into());
        self
    }

    /// Override the per-request (per-turn) timeout.
    pub fn with_request_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = request_timeout;
        self
    }
}

impl Default for ZeroclawConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ZeroclawConfig {
    /// Redacts [`token`](Self::token) and
    /// [`webhook_secret`](Self::webhook_secret); credentials must never reach
    /// a log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZeroclawConfig")
            .field("base_url", &self.base_url.as_str())
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .field(
                "webhook_secret",
                &self.webhook_secret.as_ref().map(|_| "<redacted>"),
            )
            .field("agent", &self.agent)
            .field("request_timeout", &self.request_timeout)
            .finish()
    }
}
