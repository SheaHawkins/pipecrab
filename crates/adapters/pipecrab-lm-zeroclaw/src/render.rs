//! Rendering the stage's conversation tail into the daemon prompt.
//!
//! The daemon session's history is the source of truth; the adapter reads
//! only the **last** message of the [`Conversation`] the stage hands it.
//! `LmStage`'s own conversation is bounded dead weight — see the crate docs.

use pipecrab_lm::{Conversation, LmError, Message};

/// What the conversation tail renders to.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Rendered {
    /// The prompt text for the next daemon turn.
    Prompt(String),
    /// Nothing worth a turn (whitespace-only) — the daemon rejects blank
    /// prompts, so the stage gets an immediately empty stream instead.
    Empty,
}

/// Render the last message of `conversation`:
///
/// | Last message | Prompt |
/// |---|---|
/// | `User { content }` | `content` as-is |
/// | `Event { source, kind, content }` | `[{source}/{kind}] {content}` |
/// | `ToolResult { name, content, .. }` | `[{name}] {content}` |
/// | anything else | protocol violation |
///
/// The `Event` arm is the delegation re-entry path: `DispatchIngress`
/// projects a `Completion` to `Respond(Event { source: "dispatch", .. })`,
/// and the rendering above becomes the next daemon turn.
pub(crate) fn render_last(conversation: &Conversation) -> Result<Rendered, LmError> {
    let last = conversation
        .messages
        .last()
        .ok_or_else(|| LmError::Engine("generate called on an empty conversation".into()))?;
    let prompt = match last {
        Message::User { content } => content.to_string(),
        Message::Event {
            source,
            kind,
            content,
        } => format!("[{source}/{kind}] {content}"),
        Message::ToolResult { name, content, .. } => format!("[{name}] {content}"),
        Message::System { .. } | Message::Assistant { .. } => {
            return Err(LmError::Engine(
                "conversation tail is not a user message, event, or tool result; \
                 the ZeroClaw adapter only renders re-entrant inputs"
                    .into(),
            ));
        }
    };
    if prompt.trim().is_empty() {
        return Ok(Rendered::Empty);
    }
    Ok(Rendered::Prompt(prompt))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convo(messages: Vec<Message>) -> Conversation {
        Conversation { messages }
    }

    #[test]
    fn user_text_renders_verbatim() {
        let c = convo(vec![
            Message::system("sys"),
            Message::user("turn the lights on"),
        ]);
        assert_eq!(
            render_last(&c).unwrap(),
            Rendered::Prompt("turn the lights on".into())
        );
    }

    #[test]
    fn dispatch_event_renders_bracketed() {
        let c = convo(vec![Message::Event {
            source: "dispatch".into(),
            kind: "completion".into(),
            content: "task pc-1 (agent research): done".into(),
        }]);
        assert_eq!(
            render_last(&c).unwrap(),
            Rendered::Prompt("[dispatch/completion] task pc-1 (agent research): done".into())
        );
    }

    #[test]
    fn tool_result_renders_bracketed() {
        let c = convo(vec![Message::ToolResult {
            tool_call_id: "call-1".into(),
            name: "dispatch_task".into(),
            content: "{\"task_id\":\"t1\"}".into(),
        }]);
        assert_eq!(
            render_last(&c).unwrap(),
            Rendered::Prompt("[dispatch_task] {\"task_id\":\"t1\"}".into())
        );
    }

    #[test]
    fn whitespace_only_is_empty_not_an_error() {
        let c = convo(vec![Message::user("   \n\t")]);
        assert_eq!(render_last(&c).unwrap(), Rendered::Empty);
    }

    #[test]
    fn assistant_tail_is_a_protocol_violation() {
        let c = convo(vec![Message::assistant("hello")]);
        assert!(matches!(render_last(&c), Err(LmError::Engine(_))));
    }

    #[test]
    fn empty_conversation_is_a_protocol_violation() {
        assert!(matches!(
            render_last(&convo(vec![])),
            Err(LmError::Engine(_))
        ));
    }
}
