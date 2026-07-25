//! The thin HTTP layer over the Hermes runs API: request/response wire types and
//! a [`HermesClient`] that turns a bad response into a message the caller can put
//! in a `Rejected`, or a poll into a [`PollResult`].
//!
//! The bearer token lives only in the `Authorization` header this module sets
//! per request; it is never placed in a URL, body, or error string.

use std::sync::Arc;
use std::time::Duration;

use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};
use url::Url;

use crate::config::HermesConfig;
use crate::task::{HistoryTurn, RunId};

/// How many characters of an error-response body to keep in a rejection message.
const BODY_EXCERPT_LIMIT: usize = 200;

/// The distilled result of one `GET /v1/runs/{id}`.
#[derive(Debug)]
pub(crate) enum PollResult {
    /// A 2xx run-status body.
    Status(RunView),
    /// A 404: the run is gone. For a run we are tracking, that means its
    /// terminal state expired before we observed it.
    Gone,
    /// A network error, a 5xx, a 401, or an undecodable body — the run is still
    /// real, so the poller logs `reason` and retries on the next tick.
    Transient(Arc<str>),
}

/// The fields of a run-status body this adapter acts on.
#[derive(Debug)]
pub(crate) struct RunView {
    /// The run's lifecycle status (`running`, `completed`, ...).
    pub(crate) status: String,
    /// The completed run's output, if any.
    pub(crate) output: Option<String>,
    /// Best-effort error detail for a failed run (field name undocumented, so
    /// extracted leniently — see [`RunStatusBody::error_detail`]).
    pub(crate) error: Option<String>,
}

/// The `POST /v1/runs` success body. Only the assigned `run_id` matters; the
/// echoed `status` (`"started"`) is ignored — the poller learns the real state.
#[derive(Deserialize)]
struct RunCreated {
    run_id: String,
}

/// The `GET /v1/runs/{id}` body, as much of it as the adapter reads.
#[derive(Deserialize)]
struct RunStatusBody {
    status: String,
    #[serde(default)]
    output: Option<String>,
    // The error detail field is undocumented; probe several plausible shapes so
    // a `failed` run's reason survives whatever the gateway actually names it.
    #[serde(default)]
    error: Option<Value>,
    #[serde(default)]
    last_error: Option<Value>,
    #[serde(default)]
    error_message: Option<String>,
}

impl RunStatusBody {
    fn into_view(self) -> RunView {
        RunView {
            error: self.error_detail(),
            status: self.status,
            output: self.output,
        }
    }

    /// Pull an error string out of whichever field carried it: an explicit
    /// `error_message`, then `error` / `last_error` as either a bare string or
    /// an object with a `message`.
    fn error_detail(&self) -> Option<String> {
        if let Some(msg) = &self.error_message {
            return Some(msg.clone());
        }
        value_detail(self.error.as_ref()).or_else(|| value_detail(self.last_error.as_ref()))
    }
}

/// Interpret a JSON value as an error detail: a bare string, or an object with a
/// string `message`.
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

/// Compose the `input` string a `Create` sends: the task, with any context
/// appended under a labelled heading.
pub(crate) fn compose_input(task: &str, context: Option<&str>) -> String {
    match context {
        Some(context) if !context.is_empty() => format!("{task}\n\nContext:\n{context}"),
        _ => task.to_owned(),
    }
}

/// A reqwest-backed client for the Hermes runs API. Cheap to clone (the
/// underlying [`Client`] is an `Arc`), though this adapter holds just one.
pub(crate) struct HermesClient {
    client: Client,
    base_url: Url,
    api_key: Arc<str>,
}

impl HermesClient {
    /// Build a client from `config`. The per-request timeout is baked into the
    /// underlying reqwest client.
    pub(crate) fn new(config: &HermesConfig) -> Self {
        Self {
            client: build_client(config.request_timeout),
            base_url: config.base_url.clone(),
            api_key: config.api_key.clone(),
        }
    }

    /// `POST /v1/runs`. Returns the assigned [`RunId`] on any 2xx, or a
    /// human-readable rejection message (status + body excerpt, or the transport
    /// error) that never contains the bearer token.
    pub(crate) async fn create_run(
        &self,
        input: &str,
        session_id: &str,
        idempotency_key: &str,
        history: &[HistoryTurn],
    ) -> Result<RunId, Arc<str>> {
        let mut body = json!({ "input": input, "session_id": session_id });
        // A `session_id` correlates runs but carries no conversation, so a
        // follow-up run must replay the task's turns to have any context.
        if !history.is_empty() {
            body["conversation_history"] = json!(history);
        }
        let response = self
            .client
            .post(self.runs_url())
            .bearer_auth(&*self.api_key)
            // Sent so a gateway that honors it cannot double-dispatch a retried
            // create. Gateway 0.18.2 does *not* dedupe on it (a repost with the
            // same key returns a fresh run_id), so this is belt-and-braces, not
            // a guarantee to rely on.
            .header("Idempotency-Key", idempotency_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| rejection(format!("hermes create request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            return Err(rejection(format!(
                "hermes create returned {status}: {}",
                excerpt(&response.text().await.unwrap_or_default())
            )));
        }
        let created: RunCreated = response
            .json()
            .await
            .map_err(|e| rejection(format!("hermes create returned an undecodable body: {e}")))?;
        Ok(RunId::from(created.run_id))
    }

    /// `GET /v1/runs/{run_id}`. Classifies the response into a [`PollResult`];
    /// never fails the poller — every error path is a variant, not an `Err`.
    pub(crate) async fn get_run(&self, run_id: &RunId) -> PollResult {
        let response = match self
            .client
            .get(self.run_url(run_id.as_str()))
            .bearer_auth(&*self.api_key)
            .send()
            .await
        {
            Ok(response) => response,
            Err(e) => return PollResult::Transient(format!("poll request failed: {e}").into()),
        };

        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            return PollResult::Gone;
        }
        if !status.is_success() {
            // 5xx, 401, or any other non-2xx: the run is still real, so retry.
            return PollResult::Transient(format!("poll returned {status}").into());
        }
        match response.json::<RunStatusBody>().await {
            Ok(body) => PollResult::Status(body.into_view()),
            Err(e) => {
                PollResult::Transient(format!("poll returned an undecodable body: {e}").into())
            }
        }
    }

    /// `{base}/v1/runs`, robust to a base URL with or without a trailing slash.
    fn runs_url(&self) -> Url {
        self.path(["v1", "runs"])
    }

    /// `{base}/v1/runs/{run_id}`.
    fn run_url(&self, run_id: &str) -> Url {
        self.path(["v1", "runs", run_id])
    }

    fn path<'a>(&self, segments: impl IntoIterator<Item = &'a str>) -> Url {
        let mut url = self.base_url.clone();
        url.path_segments_mut()
            .expect("base_url must be an http(s) URL that can be a base")
            .pop_if_empty()
            .extend(segments);
        url
    }
}

/// Build the reqwest client with a per-request timeout. Falls back to a default
/// client if the (very rarely failing) TLS backend init does — the poller and
/// sink will then surface transport errors normally.
fn build_client(request_timeout: Duration) -> Client {
    Client::builder()
        .timeout(request_timeout)
        .build()
        .unwrap_or_default()
}

fn rejection(message: String) -> Arc<str> {
    Arc::from(message)
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
    fn compose_input_appends_labelled_context() {
        assert_eq!(compose_input("do it", None), "do it");
        assert_eq!(compose_input("do it", Some("")), "do it");
        assert_eq!(
            compose_input("do it", Some("carefully")),
            "do it\n\nContext:\ncarefully"
        );
    }

    #[test]
    fn error_detail_reads_string_object_and_explicit_message() {
        let string = RunStatusBody {
            status: "failed".into(),
            output: None,
            error: Some(Value::String("boom".into())),
            last_error: None,
            error_message: None,
        };
        assert_eq!(string.error_detail().as_deref(), Some("boom"));

        let object = RunStatusBody {
            status: "failed".into(),
            output: None,
            error: Some(json!({ "message": "nested boom", "code": 1 })),
            last_error: None,
            error_message: None,
        };
        assert_eq!(object.error_detail().as_deref(), Some("nested boom"));

        let explicit = RunStatusBody {
            status: "failed".into(),
            output: None,
            error: None,
            last_error: Some(Value::String("ignored".into())),
            error_message: Some("explicit".into()),
        };
        assert_eq!(explicit.error_detail().as_deref(), Some("explicit"));

        let none = RunStatusBody {
            status: "failed".into(),
            output: None,
            error: None,
            last_error: None,
            error_message: None,
        };
        assert_eq!(none.error_detail(), None);
    }

    #[test]
    fn excerpt_truncates_on_a_char_boundary() {
        let long = "x".repeat(BODY_EXCERPT_LIMIT + 50);
        let out = excerpt(&long);
        assert!(out.ends_with('…'));
        assert!(out.len() <= BODY_EXCERPT_LIMIT + 4);
    }
}
