//! Hand-mirrored subset of the ZeroClaw daemon's JSON-RPC wire protocol.
//!
//! Mirroring instead of importing trades compile-time coupling for
//! wire-compatibility risk: this crate carries no ZeroClaw dependency, and the
//! ignored integration test is the tripwire for drift. Everything on the wire
//! is snake_case (ZeroClaw's `rpc_type!` macro applies it globally), messages
//! are newline-delimited JSON-RPC 2.0, and `session/update` notifications are
//! a `"type"`-tagged enum.
//!
//! Updates are classified by hand rather than derived: an unknown update kind
//! must degrade to [`SessionUpdate::Unknown`], never to a deserialization
//! error that kills the turn.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC method names this adapter speaks.
pub(crate) mod method {
    pub(crate) const INITIALIZE: &str = "initialize";
    pub(crate) const SESSION_NEW: &str = "session/new";
    pub(crate) const SESSION_PROMPT: &str = "session/prompt";
    pub(crate) const SESSION_CANCEL: &str = "session/cancel";
    pub(crate) const SESSION_UPDATE: &str = "session/update";
}

/// `initialize` request parameters. `tui_id`/`tui_sig` echo a previous
/// connection's assigned identity so a reconnect is recognized.
#[derive(Debug, Serialize)]
pub(crate) struct InitializeParams {
    pub(crate) protocol_version: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tui_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tui_sig: Option<String>,
}

/// `initialize` result: the daemon's assigned identity for this client.
#[derive(Debug, Deserialize)]
pub(crate) struct InitializeResult {
    #[serde(default)]
    pub(crate) tui_id: Option<String>,
    #[serde(default)]
    pub(crate) tui_sig: Option<String>,
}

/// `session/new` request parameters. Supplying `session_id` reattaches to an
/// existing session; `chat_mode` is always `"chat"` for this adapter.
#[derive(Debug, Serialize)]
pub(crate) struct SessionNewParams {
    pub(crate) agent_alias: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) exclude_memory: Option<bool>,
    pub(crate) chat_mode: &'static str,
}

/// `session/new` result. `message_count` is nonzero when the daemon
/// rehydrated an existing session; `workspace_dir` locates the delegation
/// results the poller watches.
#[derive(Debug, Deserialize)]
pub(crate) struct SessionNewResult {
    pub(crate) session_id: String,
    #[serde(default)]
    pub(crate) message_count: usize,
    pub(crate) workspace_dir: String,
}

/// `session/prompt` request parameters. `attachments` is omitted — the
/// daemon defaults it to empty.
#[derive(Debug, Serialize)]
pub(crate) struct SessionPromptParams {
    pub(crate) session_id: String,
    pub(crate) prompt: String,
}

/// `session/prompt` result. Arrives after the terminal
/// [`SessionUpdate::TurnComplete`] on the same ordered socket; the adapter
/// treats it as a fallback terminal only.
#[derive(Debug, Deserialize)]
pub(crate) struct SessionPromptResult {
    #[serde(default)]
    pub(crate) content: String,
}

/// `session/cancel` request parameters.
#[derive(Debug, Serialize)]
pub(crate) struct SessionIdParams {
    pub(crate) session_id: String,
}

/// A JSON-RPC error object, as returned in a response's `error` member.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RpcErrorObject {
    #[serde(default)]
    pub(crate) code: i64,
    pub(crate) message: String,
}

impl std::fmt::Display for RpcErrorObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (code {})", self.message, self.code)
    }
}

/// How one streamed `session/prompt` turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnOutcome {
    Completed,
    Cancelled,
    Failed,
}

/// One `session/update` notification, classified.
///
/// Only the fields this adapter consumes are extracted; everything else in
/// the payload is ignored. Unknown `type` tags become [`Unknown`]
/// (forward-compatible), and a missing/foreign `session_id` is the caller's
/// problem to filter — every variant carries it.
///
/// [`Unknown`]: SessionUpdate::Unknown
#[derive(Debug, Clone)]
pub(crate) enum SessionUpdate {
    MessageChunk {
        session_id: String,
        text: String,
    },
    ThoughtChunk {
        session_id: String,
    },
    ToolCall {
        session_id: String,
        tool_call_id: String,
        name: String,
        raw_input: Value,
    },
    ToolResult {
        session_id: String,
    },
    ApprovalRequest {
        session_id: String,
        tool_name: String,
    },
    ContextUsage {
        session_id: String,
    },
    Plan {
        session_id: String,
    },
    HistoryTrimmed {
        session_id: String,
        dropped_messages: u64,
        reason: String,
    },
    TurnComplete {
        session_id: String,
        outcome: TurnOutcome,
        content: String,
    },
    Unknown {
        session_id: Option<String>,
    },
}

impl SessionUpdate {
    /// The session this update belongs to, when it named one.
    pub(crate) fn session_id(&self) -> Option<&str> {
        match self {
            SessionUpdate::MessageChunk { session_id, .. }
            | SessionUpdate::ThoughtChunk { session_id }
            | SessionUpdate::ToolCall { session_id, .. }
            | SessionUpdate::ToolResult { session_id }
            | SessionUpdate::ApprovalRequest { session_id, .. }
            | SessionUpdate::ContextUsage { session_id }
            | SessionUpdate::Plan { session_id }
            | SessionUpdate::HistoryTrimmed { session_id, .. }
            | SessionUpdate::TurnComplete { session_id, .. } => Some(session_id),
            SessionUpdate::Unknown { session_id } => session_id.as_deref(),
        }
    }

    /// Classify a `session/update` notification's `params` value.
    pub(crate) fn from_params(params: &Value) -> SessionUpdate {
        fn text(params: &Value, key: &str) -> String {
            params
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        }

        let session = params.get("session_id").and_then(Value::as_str);
        let Some(session_id) = session.map(str::to_owned) else {
            return SessionUpdate::Unknown { session_id: None };
        };
        let kind = params.get("type").and_then(Value::as_str).unwrap_or("");

        match kind {
            "agent_message_chunk" => SessionUpdate::MessageChunk {
                session_id,
                text: text(params, "text"),
            },
            "agent_thought_chunk" => SessionUpdate::ThoughtChunk { session_id },
            "tool_call" => SessionUpdate::ToolCall {
                session_id,
                tool_call_id: text(params, "tool_call_id"),
                name: text(params, "name"),
                raw_input: params.get("raw_input").cloned().unwrap_or(Value::Null),
            },
            "tool_result" => SessionUpdate::ToolResult { session_id },
            "approval_request" => SessionUpdate::ApprovalRequest {
                session_id,
                tool_name: text(params, "tool_name"),
            },
            "context_usage" => SessionUpdate::ContextUsage { session_id },
            "plan" => SessionUpdate::Plan { session_id },
            "history_trimmed" => SessionUpdate::HistoryTrimmed {
                session_id,
                dropped_messages: params
                    .get("dropped_messages")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                reason: text(params, "reason"),
            },
            "turn_complete" => {
                let outcome = match params.get("outcome").and_then(Value::as_str) {
                    Some("completed") => TurnOutcome::Completed,
                    Some("cancelled") => TurnOutcome::Cancelled,
                    // An unknown outcome is a failure: claiming completion on
                    // an unrecognized terminal would fabricate an empty reply.
                    _ => TurnOutcome::Failed,
                };
                SessionUpdate::TurnComplete {
                    session_id,
                    outcome,
                    content: text(params, "content"),
                }
            }
            _ => SessionUpdate::Unknown {
                session_id: Some(session_id),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classifies_message_chunk() {
        let update = SessionUpdate::from_params(&json!({
            "type": "agent_message_chunk", "session_id": "s1", "text": "hi",
        }));
        let SessionUpdate::MessageChunk { session_id, text } = update else {
            panic!("expected MessageChunk, got {update:?}");
        };
        assert_eq!(session_id, "s1");
        assert_eq!(text, "hi");
    }

    #[test]
    fn classifies_turn_complete_outcomes() {
        for (wire, expected) in [
            ("completed", TurnOutcome::Completed),
            ("cancelled", TurnOutcome::Cancelled),
            ("failed", TurnOutcome::Failed),
            ("some_future_outcome", TurnOutcome::Failed),
        ] {
            let update = SessionUpdate::from_params(&json!({
                "type": "turn_complete", "session_id": "s1",
                "outcome": wire, "content": "c",
            }));
            let SessionUpdate::TurnComplete { outcome, .. } = update else {
                panic!("expected TurnComplete, got {update:?}");
            };
            assert_eq!(outcome, expected, "wire outcome {wire:?}");
        }
    }

    #[test]
    fn unknown_type_and_missing_session_degrade_gracefully() {
        let unknown = SessionUpdate::from_params(&json!({
            "type": "brand_new_event", "session_id": "s1",
        }));
        assert!(matches!(
            unknown,
            SessionUpdate::Unknown { session_id: Some(ref s) } if s == "s1"
        ));

        let sessionless = SessionUpdate::from_params(&json!({ "type": "plan" }));
        assert!(matches!(
            sessionless,
            SessionUpdate::Unknown { session_id: None }
        ));
    }

    #[test]
    fn params_serialize_snake_case_and_skip_absent_options() {
        let params = serde_json::to_value(SessionNewParams {
            agent_alias: "voice".into(),
            session_id: Some("pc-voice-1".into()),
            exclude_memory: None,
            chat_mode: "chat",
        })
        .unwrap();
        assert_eq!(
            params,
            json!({
                "agent_alias": "voice",
                "session_id": "pc-voice-1",
                "chat_mode": "chat",
            })
        );

        let init = serde_json::to_value(InitializeParams {
            protocol_version: 1,
            tui_id: None,
            tui_sig: None,
        })
        .unwrap();
        assert_eq!(init, json!({ "protocol_version": 1 }));
    }
}
