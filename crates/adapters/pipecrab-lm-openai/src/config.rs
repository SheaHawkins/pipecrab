//! [`OpenAiConfig`]: which endpoint to reach, as whom, and the per-host
//! adjustments that let one adapter serve every OpenAI-compatible provider.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Map, Value};
use url::Url;

/// OpenAI's own endpoint, including the version segment.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// How long to wait for the TCP/TLS connection.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the body may go silent between chunks. Deliberately not a whole-
/// request deadline: a long healthy generation must not be aborted for taking
/// its time, so what is bounded is *silence*.
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Connection parameters for an OpenAI-compatible Chat Completions endpoint.
///
/// Construct with [`OpenAiConfig::new`] and override with the builder methods,
/// or fill the public fields directly.
///
/// The [`api_key`](Self::api_key) is a secret: [`Debug`] redacts it, it is set
/// as a sensitive header value, and it appears in no URL, body, or error
/// message. An **empty** key omits the `Authorization` header entirely, which is
/// how a local `llama-server` or Ollama endpoint is reached.
///
/// [`base_url`](Self::base_url) is taken literally and should carry the version
/// segment (`https://openrouter.ai/api/v1`, `http://127.0.0.1:8080/v1`). The
/// adapter appends `chat/completions` to it; a trailing slash is harmless and no
/// `/v1` is inferred.
#[derive(Clone)]
pub struct OpenAiConfig {
    /// Endpoint base URL, version segment included. Defaults to
    /// `https://api.openai.com/v1`.
    pub base_url: Url,
    /// Bearer token. Redacted from [`Debug`]; empty omits the header.
    pub api_key: Arc<str>,
    /// The model name sent with every request.
    pub model: Arc<str>,
    /// Connection timeout. Defaults to 10s.
    pub connect_timeout: Duration,
    /// Maximum silence between response-body chunks. Defaults to 30s.
    pub read_timeout: Duration,
    /// Extra request headers, applied after the `Authorization` header.
    pub headers: Vec<(String, String)>,
    /// Fields merged into the request body, last, so one may also correct a
    /// field the adapter sets.
    pub extra_body: Map<String, Value>,
    /// Whether to stream the response. Defaults to `true`.
    pub streaming: bool,
}

impl OpenAiConfig {
    /// A config for `model` against OpenAI's endpoint, authenticated with
    /// `api_key`. Pass an empty key for a host that wants none.
    pub fn new(model: impl Into<Arc<str>>, api_key: impl Into<Arc<str>>) -> Self {
        Self {
            base_url: Url::parse(DEFAULT_BASE_URL).expect("the default base URL is valid"),
            api_key: api_key.into(),
            model: model.into(),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            read_timeout: DEFAULT_READ_TIMEOUT,
            headers: Vec::new(),
            extra_body: Map::new(),
            streaming: true,
        }
    }

    /// Point at another OpenAI-compatible host.
    pub fn with_base_url(mut self, base_url: Url) -> Self {
        self.base_url = base_url;
        self
    }

    /// Override the connection timeout.
    pub fn with_connect_timeout(mut self, connect_timeout: Duration) -> Self {
        self.connect_timeout = connect_timeout;
        self
    }

    /// Override how long the response body may go silent.
    pub fn with_read_timeout(mut self, read_timeout: Duration) -> Self {
        self.read_timeout = read_timeout;
        self
    }

    /// Add a request header — OpenRouter's `HTTP-Referer` / `X-Title`, Azure's
    /// `api-key`, a proxy's tenant header. Repeatable.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Merge fields into the request body — `top_p`, OpenRouter's `provider`
    /// routing, a host that wants `max_completion_tokens` instead of
    /// `max_tokens`. Applied last, so a field the adapter sets can be corrected.
    pub fn with_extra_body(mut self, extra_body: Map<String, Value>) -> Self {
        self.extra_body.extend(extra_body);
        self
    }

    /// Turn streaming off, trading a turn's latency for compatibility with a
    /// host whose SSE tool calls are broken but whose blocking endpoint is fine.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }
}

impl fmt::Debug for OpenAiConfig {
    /// Redacts [`api_key`](Self::api_key); the token must never reach a log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiConfig")
            .field("base_url", &self.base_url.as_str())
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("connect_timeout", &self.connect_timeout)
            .field("read_timeout", &self.read_timeout)
            .field("headers", &self.headers)
            .field("extra_body", &self.extra_body)
            .field("streaming", &self.streaming)
            .finish()
    }
}

/// Why an [`OpenAi`](crate::OpenAi) handle could not be built. Every variant is
/// a static configuration fault, surfaced before any request.
#[derive(Debug, thiserror::Error)]
pub enum OpenAiBuildError {
    /// The model name was empty.
    #[error("model name must be nonempty")]
    EmptyModel,
    /// The base URL cannot have path segments appended (a `mailto:`-style URL).
    #[error("base URL must be an http(s) URL that can be a base: {0}")]
    BaseUrl(Url),
    /// A header name or value is not valid HTTP. The value is not echoed — it
    /// may itself be a credential.
    #[error("invalid header {name:?}: {reason}")]
    Header {
        /// The offending header's name.
        name: String,
        /// What was wrong with it.
        reason: String,
    },
    /// The HTTP client could not be constructed.
    #[error("failed to build the HTTP client: {0}")]
    Client(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_the_api_key() {
        let config = OpenAiConfig::new("gpt-5", "sk-super-secret");
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("sk-super-secret"), "{rendered}");
        assert!(rendered.contains("<redacted>"), "{rendered}");
    }

    #[test]
    fn defaults_target_openai_with_streaming_on() {
        let config = OpenAiConfig::new("gpt-5", "");
        assert_eq!(config.base_url.as_str(), "https://api.openai.com/v1");
        assert!(config.streaming);
        assert!(config.headers.is_empty());
        assert!(config.extra_body.is_empty());
    }

    #[test]
    fn extra_body_accumulates_across_calls() {
        let mut first = Map::new();
        first.insert("top_p".into(), Value::from(0.5));
        let mut second = Map::new();
        second.insert("seed".into(), Value::from(7));

        let config = OpenAiConfig::new("gpt-5", "")
            .with_extra_body(first)
            .with_extra_body(second);

        assert_eq!(config.extra_body.get("top_p"), Some(&Value::from(0.5)));
        assert_eq!(config.extra_body.get("seed"), Some(&Value::from(7)));
    }
}
