//! The reqwest edge: one validated endpoint, the headers every request carries,
//! and the mapping from a bad response to an [`LmError`].
//!
//! The bearer token lives only in the `Authorization` header built here, marked
//! sensitive so it stays out of logs. It reaches no URL, body, or error string.

use std::sync::Arc;

use pipecrab_lm::{Conversation, GenParams, LmError, ToolDefinition};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, Response};
use serde_json::{Map, Value};
use url::Url;

use crate::config::{OpenAiBuildError, OpenAiConfig};
use crate::wire;

/// How many characters of an error-response body to keep in an error message.
const BODY_EXCERPT_LIMIT: usize = 200;

/// A client bound to one Chat Completions endpoint.
pub(crate) struct OpenAiClient {
    client: Client,
    endpoint: Url,
    headers: HeaderMap,
    model: Arc<str>,
    streaming: bool,
    extra_body: Map<String, Value>,
}

impl OpenAiClient {
    /// Resolve `config` into an endpoint, a header set, and an HTTP client.
    /// Every fault here is static, so a built client cannot fail for
    /// configuration reasons later.
    pub(crate) fn new(config: &OpenAiConfig) -> Result<Self, OpenAiBuildError> {
        if config.model.is_empty() {
            return Err(OpenAiBuildError::EmptyModel);
        }
        let client = Client::builder()
            .connect_timeout(config.connect_timeout)
            // Bounds silence between body chunks rather than the whole request:
            // a long healthy generation must not be aborted for taking its time.
            .read_timeout(config.read_timeout)
            .build()
            .map_err(|error| OpenAiBuildError::Client(error.to_string()))?;
        Ok(Self {
            client,
            endpoint: endpoint(&config.base_url)?,
            headers: headers(config)?,
            model: config.model.clone(),
            streaming: config.streaming,
            extra_body: config.extra_body.clone(),
        })
    }

    /// Whether responses are streamed.
    pub(crate) fn streaming(&self) -> bool {
        self.streaming
    }

    /// Render one generation's request body.
    pub(crate) fn request_body(
        &self,
        conversation: &Conversation,
        params: &GenParams,
        tools: &[ToolDefinition],
    ) -> Result<Value, LmError> {
        wire::build_request(
            &self.model,
            conversation,
            params,
            tools,
            self.streaming,
            &self.extra_body,
        )
    }

    /// Post `body`, returning the response with its body unread.
    pub(crate) async fn send(&self, body: Value) -> Result<Response, LmError> {
        let response = self
            .client
            .post(self.endpoint.clone())
            .headers(self.headers.clone())
            .json(&body)
            .send()
            .await
            .map_err(|error| LmError::Engine(format!("completion request failed: {error}")))?;

        let status = response.status();
        if !status.is_success() {
            // The status is kept in the message so 401, 429, and 500 stay
            // distinguishable in a log without a variant per code.
            let body = response.text().await.unwrap_or_default();
            return Err(LmError::Engine(format!(
                "completion request returned {status}: {}",
                excerpt(&body)
            )));
        }
        Ok(response)
    }
}

/// `{base}/chat/completions`, robust to a base URL with or without a trailing
/// slash. The base is taken literally — no version segment is inferred.
fn endpoint(base_url: &Url) -> Result<Url, OpenAiBuildError> {
    let mut url = base_url.clone();
    url.path_segments_mut()
        .map_err(|()| OpenAiBuildError::BaseUrl(base_url.clone()))?
        .pop_if_empty()
        .extend(["chat", "completions"]);
    Ok(url)
}

/// The headers every request carries: the bearer token when there is one, then
/// the caller's own, which may therefore override it.
fn headers(config: &OpenAiConfig) -> Result<HeaderMap, OpenAiBuildError> {
    let mut headers = HeaderMap::new();
    // An empty key means the host wants none — a local llama-server or Ollama.
    if !config.api_key.is_empty() {
        let mut value =
            HeaderValue::from_str(&format!("Bearer {}", config.api_key)).map_err(|_| {
                OpenAiBuildError::Header {
                    name: AUTHORIZATION.to_string(),
                    reason: "the API key is not a valid header value".into(),
                }
            })?;
        // Keeps the token out of hyper's wire logs.
        value.set_sensitive(true);
        headers.insert(AUTHORIZATION, value);
    }
    for (name, value) in &config.headers {
        let header =
            HeaderName::from_bytes(name.as_bytes()).map_err(|error| OpenAiBuildError::Header {
                name: name.clone(),
                reason: error.to_string(),
            })?;
        // The value is never echoed into the error: it may itself be a secret.
        let value = HeaderValue::from_str(value).map_err(|_| OpenAiBuildError::Header {
            name: name.clone(),
            reason: "value is not valid for an HTTP header".into(),
        })?;
        headers.insert(header, value);
    }
    Ok(headers)
}

/// Truncate a response body for inclusion in an error message.
fn excerpt(body: &str) -> String {
    let body = body.trim();
    if body.len() <= BODY_EXCERPT_LIMIT {
        return body.to_owned();
    }
    // Truncate on a char boundary at or below the limit.
    let end = body
        .char_indices()
        .take_while(|(index, _)| *index <= BODY_EXCERPT_LIMIT)
        .last()
        .map(|(index, _)| index)
        .unwrap_or(0);
    format!("{}…", &body[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(base_url: &str) -> OpenAiConfig {
        OpenAiConfig::new("gpt-5", "sk-secret").with_base_url(Url::parse(base_url).expect("a URL"))
    }

    #[test]
    fn the_endpoint_appends_to_the_base_path() {
        assert_eq!(
            endpoint(&Url::parse("https://openrouter.ai/api/v1").unwrap())
                .unwrap()
                .as_str(),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }

    #[test]
    fn a_trailing_slash_does_not_double_up() {
        assert_eq!(
            endpoint(&Url::parse("http://127.0.0.1:8080/v1/").unwrap())
                .unwrap()
                .as_str(),
            "http://127.0.0.1:8080/v1/chat/completions"
        );
    }

    #[test]
    fn a_url_that_cannot_be_a_base_is_rejected() {
        let error = endpoint(&Url::parse("mailto:someone@example.com").unwrap()).unwrap_err();
        assert!(matches!(error, OpenAiBuildError::BaseUrl(_)), "{error:?}");
    }

    #[test]
    fn the_api_key_becomes_a_sensitive_bearer_header() {
        let headers = headers(&config("https://api.openai.com/v1")).unwrap();
        let value = headers.get(AUTHORIZATION).expect("an authorization header");
        assert!(value.is_sensitive());
        assert_eq!(value.to_str().unwrap(), "Bearer sk-secret");
    }

    #[test]
    fn an_empty_key_omits_the_authorization_header() {
        let config = OpenAiConfig::new("qwen3", "");
        assert!(headers(&config).unwrap().get(AUTHORIZATION).is_none());
    }

    #[test]
    fn extra_headers_are_applied() {
        let config = config("https://openrouter.ai/api/v1")
            .with_header("HTTP-Referer", "https://example.com")
            .with_header("X-Title", "pipecrab");
        let headers = headers(&config).unwrap();
        assert_eq!(headers.get("http-referer").unwrap(), "https://example.com");
        assert_eq!(headers.get("x-title").unwrap(), "pipecrab");
    }

    #[test]
    fn an_invalid_header_name_is_a_build_error() {
        let config = config("https://api.openai.com/v1").with_header("bad header", "x");
        let error = headers(&config).unwrap_err();
        assert!(
            matches!(&error, OpenAiBuildError::Header { name, .. } if name == "bad header"),
            "{error:?}"
        );
    }

    #[test]
    fn an_invalid_header_value_does_not_echo_the_value() {
        let config = config("https://api.openai.com/v1").with_header("x-tenant", "line\nbreak");
        let error = headers(&config).unwrap_err();
        assert!(!error.to_string().contains("line"), "{error}");
    }

    #[test]
    fn an_empty_model_is_rejected_before_any_request() {
        let config = OpenAiConfig::new("", "sk-secret");
        assert!(matches!(
            OpenAiClient::new(&config),
            Err(OpenAiBuildError::EmptyModel)
        ));
    }

    #[test]
    fn excerpt_truncates_on_a_char_boundary() {
        let long = "x".repeat(BODY_EXCERPT_LIMIT + 50);
        let out = excerpt(&long);
        assert!(out.ends_with('…'));
        assert!(out.len() <= BODY_EXCERPT_LIMIT + 4);
    }
}
