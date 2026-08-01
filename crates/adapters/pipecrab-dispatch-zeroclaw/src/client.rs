//! The thin HTTP layer over the ZeroClaw webhook API: one POST per agent turn,
//! its reply parsed leniently into a [`PostResult`] for [`classify`] to judge.
//!
//! The bearer token and webhook secret live only in the headers this module
//! sets per request; neither is ever placed in a URL, body, or error string.
//!
//! [`classify`]: crate::classify::classify

use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use url::Url;

use crate::config::ZeroclawConfig;
use crate::task::Turn;

/// How many characters of a response body to keep in a failure message.
const BODY_EXCERPT_LIMIT: usize = 200;

/// The outcome of one `POST /webhook`, however it went.
#[derive(Debug)]
pub(crate) enum PostResult {
    /// The gateway answered; `status` plus whatever the body carried.
    Response {
        /// The HTTP status code.
        status: u16,
        /// The leniently parsed body.
        body: ReplyBody,
    },
    /// The request never completed: a connect error, a timeout, a dropped
    /// connection. The turn may or may not have run gateway-side.
    Transport {
        /// The transport error, rendered.
        message: String,
    },
}

/// The fields of a webhook reply this adapter acts on, whatever the status.
/// A non-JSON body degrades to an [`error`](Self::error) excerpt rather than
/// failing to parse.
#[derive(Debug, Default)]
pub(crate) struct ReplyBody {
    /// The agent's reply (`response` on a 2xx).
    pub(crate) response: Option<String>,
    /// The error detail (`error` on a non-2xx, or a body excerpt).
    pub(crate) error: Option<String>,
    /// The dedupe marker: `"duplicate"` when an `X-Idempotency-Key` was
    /// already processed.
    pub(crate) status: Option<String>,
}

/// The webhook body shapes the gateway sends, as much as the adapter reads.
#[derive(Deserialize)]
struct ReplyWire {
    #[serde(default)]
    response: Option<String>,
    // Observed as a bare string; parsed leniently in case a build nests it.
    #[serde(default)]
    error: Option<Value>,
    #[serde(default)]
    status: Option<String>,
}

impl ReplyBody {
    /// Parse a body's text: JSON when it is, an error excerpt when it isn't.
    fn parse(text: &str) -> Self {
        match serde_json::from_str::<ReplyWire>(text) {
            Ok(wire) => Self {
                response: wire.response,
                error: value_detail(wire.error.as_ref()),
                status: wire.status,
            },
            Err(_) => Self {
                response: None,
                error: Some(excerpt(text)),
                status: None,
            },
        }
    }
}

/// Interpret a JSON value as an error detail: a bare string, or an object with
/// a string `message`.
fn value_detail(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.clone()),
        Value::Object(map) => map
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

/// Compose the message a `Create` sends: the task, with any context appended
/// under a labelled heading.
pub(crate) fn compose_message(task: &str, context: Option<&str>) -> String {
    match context {
        Some(context) if !context.is_empty() => format!("{task}\n\nContext:\n{context}"),
        _ => task.to_owned(),
    }
}

/// Compose the message an `Update` sends: the task's prior exchanges replayed
/// as a transcript, then the new message.
///
/// `POST /webhook` starts a fresh conversation every time — `X-Session-Id` is
/// not read by the gateway — so without this replay a follow-up arrives with no
/// idea what it is following up on. An empty `transcript` (no turn has yet
/// completed) sends the message alone.
pub(crate) fn compose_follow_up(transcript: &[Turn], message: &str) -> String {
    if transcript.is_empty() {
        return message.to_owned();
    }
    let mut out = String::from(
        "You are continuing an earlier conversation. \
         The transcript so far:\n\n",
    );
    for turn in transcript {
        out.push_str("User:\n");
        out.push_str(&turn.user);
        out.push_str("\n\nYou:\n");
        out.push_str(&turn.agent);
        out.push_str("\n\n");
    }
    out.push_str("The user's new message:\n\n");
    out.push_str(message);
    out
}

/// A reqwest-backed client for the ZeroClaw webhook API. Cheap to clone (the
/// underlying [`Client`] is an `Arc`), though this adapter holds just one.
pub(crate) struct ZeroclawClient {
    client: Client,
    base_url: Url,
    token: Option<Arc<str>>,
    webhook_secret: Option<Arc<str>>,
    agent: Option<Arc<str>>,
}

impl ZeroclawClient {
    /// Build a client from `config`. The per-request timeout is baked into the
    /// underlying reqwest client.
    pub(crate) fn new(config: &ZeroclawConfig) -> Self {
        Self {
            client: build_client(config.request_timeout),
            base_url: config.base_url.clone(),
            token: config.token.clone(),
            webhook_secret: config.webhook_secret.clone(),
            agent: config.agent.clone(),
        }
    }

    /// `POST /webhook`, blocking for the whole agent turn. Every path is a
    /// [`PostResult`] variant, never an `Err` — classification is the caller's.
    pub(crate) async fn post_message(
        &self,
        session_id: &str,
        message: &str,
        idempotency_key: &str,
    ) -> PostResult {
        let mut request = self
            .client
            .post(self.webhook_url())
            // Honored by the gateway: a retried key answers a `duplicate`
            // body instead of re-running the turn.
            .header("X-Idempotency-Key", idempotency_key)
            // Keys the gateway's session memory; how a follow-up post
            // continues this task's conversation.
            .header("X-Session-Id", session_id)
            .json(&serde_json::json!({ "message": message }));
        if let Some(token) = &self.token {
            request = request.bearer_auth(&**token);
        }
        if let Some(secret) = &self.webhook_secret {
            request = request.header("X-Webhook-Secret", &**secret);
        }

        let response = match request.send().await {
            Ok(response) => response,
            Err(e) => {
                return PostResult::Transport {
                    message: format!("zeroclaw webhook request failed: {e}"),
                };
            }
        };

        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        PostResult::Response {
            status,
            body: ReplyBody::parse(&text),
        }
    }

    /// `{base}/webhook`, robust to a base URL with or without a trailing
    /// slash, with the agent alias attached when configured.
    fn webhook_url(&self) -> Url {
        let mut url = self.base_url.clone();
        url.path_segments_mut()
            .expect("base_url must be an http(s) URL that can be a base")
            .pop_if_empty()
            .push("webhook");
        if let Some(agent) = &self.agent {
            url.query_pairs_mut().append_pair("agent", agent);
        }
        url
    }
}

/// Build the reqwest client with a per-request timeout. Falls back to a
/// default client if the (very rarely failing) TLS backend init does — the
/// workers will then surface transport errors normally.
fn build_client(request_timeout: Duration) -> Client {
    Client::builder()
        .timeout(request_timeout)
        .build()
        .unwrap_or_default()
}

/// Truncate a response body for inclusion in a failure message.
fn excerpt(body: &str) -> String {
    let body = body.trim();
    if body.len() <= BODY_EXCERPT_LIMIT {
        return body.to_owned();
    }
    // Truncate on a char boundary at or below the limit.
    let end = body
        .char_indices()
        .take_while(|(i, _)| *i <= BODY_EXCERPT_LIMIT)
        .last()
        .map(|(i, _)| i)
        .unwrap_or(0);
    format!("{}…", &body[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_message_appends_labelled_context() {
        assert_eq!(compose_message("do it", None), "do it");
        assert_eq!(compose_message("do it", Some("")), "do it");
        assert_eq!(
            compose_message("do it", Some("carefully")),
            "do it\n\nContext:\ncarefully"
        );
    }

    #[test]
    fn compose_follow_up_replays_the_transcript_before_the_new_message() {
        let turn = |user: &str, agent: &str| Turn {
            user: user.to_owned(),
            agent: agent.to_owned(),
        };

        // No completed exchange yet: the message travels alone, so a follow-up
        // never fabricates a conversation that did not happen.
        assert_eq!(compose_follow_up(&[], "just this"), "just this");

        let out = compose_follow_up(
            &[turn("what is 2+2?", "4"), turn("and doubled?", "8")],
            "and again?",
        );
        for fragment in ["what is 2+2?", "4", "and doubled?", "8", "and again?"] {
            assert!(out.contains(fragment), "missing {fragment:?} in: {out}");
        }
        // Order carries the meaning: history first, oldest to newest.
        assert!(out.find("what is 2+2?") < out.find("and doubled?"));
        assert!(out.find("and doubled?") < out.find("and again?"));
    }

    #[test]
    fn reply_body_parses_the_observed_shapes() {
        let ok = ReplyBody::parse(r#"{"response": "hi", "model": "glm-5"}"#);
        assert_eq!(ok.response.as_deref(), Some("hi"));
        assert_eq!(ok.error, None);

        let err = ReplyBody::parse(r#"{"error": "LLM request failed"}"#);
        assert_eq!(err.error.as_deref(), Some("LLM request failed"));

        let nested = ReplyBody::parse(r#"{"error": {"message": "nested", "code": 1}}"#);
        assert_eq!(nested.error.as_deref(), Some("nested"));

        let dup = ReplyBody::parse(
            r#"{"status": "duplicate", "idempotent": true, "message": "already processed"}"#,
        );
        assert_eq!(dup.status.as_deref(), Some("duplicate"));
    }

    #[test]
    fn a_non_json_body_degrades_to_an_error_excerpt() {
        let body = ReplyBody::parse("<html>Bad Gateway</html>");
        assert_eq!(body.response, None);
        assert_eq!(body.error.as_deref(), Some("<html>Bad Gateway</html>"));
    }

    #[test]
    fn excerpt_truncates_on_a_char_boundary() {
        let long = "x".repeat(BODY_EXCERPT_LIMIT + 50);
        let out = excerpt(&long);
        assert!(out.ends_with('…'));
        assert!(out.len() <= BODY_EXCERPT_LIMIT + 4);
    }

    #[test]
    fn webhook_url_survives_trailing_slashes_and_carries_the_agent() {
        let config = ZeroclawConfig::new()
            .with_base_url(Url::parse("http://gw.local:42617/prefix/").unwrap())
            .with_agent("researcher");
        let client = ZeroclawClient::new(&config);
        assert_eq!(
            client.webhook_url().as_str(),
            "http://gw.local:42617/prefix/webhook?agent=researcher"
        );

        let bare = ZeroclawClient::new(&ZeroclawConfig::new());
        assert_eq!(
            bare.webhook_url().as_str(),
            "http://127.0.0.1:42617/webhook"
        );
    }
}
