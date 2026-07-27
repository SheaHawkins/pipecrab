//! Tool-calling conventions: how a model family declares tools in the prompt,
//! delimits a call on the way out, and reads one back.

use std::sync::Arc;

use pipecrab_lm::{LmError, ToolCall, ToolDefinition};

/// Name of the GBNF entry rule a [`ToolDialect::grammar`] must define.
pub const TOOL_CALL_ROOT: &str = "tool-call";

/// One model family's tool-calling convention.
///
/// `llama-cpp-2` wraps the legacy `llama_chat_apply_template`, which takes only
/// role/content pairs — no tools, no template kwargs — so a dialect renders its
/// blocks *into* the content strings and parses calls back out of the token
/// stream itself.
pub trait ToolDialect: std::fmt::Debug + Send + Sync {
    /// GGUF `general.name` fragments of the models this dialect covers, matched
    /// case-insensitively by [`supports`](Self::supports).
    fn supported_models(&self) -> &'static [&'static str];

    /// Whether `model_name` — a GGUF `general.name` — is one this dialect covers.
    fn supports(&self, model_name: &str) -> bool {
        let model_name = model_name.to_lowercase();
        self.supported_models()
            .iter()
            .any(|supported| model_name.contains(&supported.to_lowercase()))
    }

    /// Tool declarations, appended to the system message.
    fn declarations(&self, tools: &[ToolDefinition]) -> String;

    /// An assistant turn's calls, appended to its visible text.
    fn tool_calls(&self, calls: &[ToolCall]) -> String;

    /// A tool result as a role the chat template knows, plus its content.
    fn tool_result(&self, content: &str) -> (&'static str, String);

    /// Literal that opens a call: the lazy-grammar trigger and the streaming
    /// withhold marker.
    fn open(&self) -> &str;

    /// Literal that closes a call.
    fn close(&self) -> &str;

    /// GBNF constraining a call once triggered.
    ///
    /// Must define [`TOOL_CALL_ROOT`], and that rule must start with
    /// [`open`](Self::open): a lazy trigger feeds the grammar from the start of
    /// the match, delimiter included.
    fn grammar(&self, tools: &[ToolDefinition]) -> Result<String, LmError>;

    /// Read a captured call body — the text between [`open`](Self::open) and
    /// [`close`](Self::close) — as a tool name and its arguments.
    fn parse(&self, body: &str) -> Result<(Arc<str>, serde_json::Value), LmError>;
}

/// `<tool_call>{"name": …, "arguments": …}</tool_call>`, with declarations in a
/// `<tools>` block.
///
/// Qwen 2.5/3 and the Nous Hermes models; llama.cpp calls it `HERMES_2_PRO`.
/// The rendering reproduces Qwen's own chat template, so the model sees the
/// token sequence it was trained on.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChatMlXml;

impl ChatMlXml {
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";
}

impl ToolDialect for ChatMlXml {
    fn supported_models(&self) -> &'static [&'static str] {
        &["Qwen", "Hermes"]
    }

    fn declarations(&self, tools: &[ToolDefinition]) -> String {
        let mut block = String::from(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        );
        for tool in tools {
            block.push_str(&format!(
                "\n{{\"type\": \"function\", \"function\": {{\"name\": {}, \"description\": {}, \
                 \"parameters\": {}}}}}",
                quoted(&tool.name),
                quoted(&tool.description),
                tool.parameters,
            ));
        }
        block.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function name and \
             arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
        );
        block
    }

    fn tool_calls(&self, calls: &[ToolCall]) -> String {
        let mut rendered = String::new();
        for call in calls {
            rendered.push_str(&format!(
                "\n<tool_call>\n{{\"name\": {}, \"arguments\": {}}}\n</tool_call>",
                quoted(&call.name),
                call.arguments_json,
            ));
        }
        rendered
    }

    fn tool_result(&self, content: &str) -> (&'static str, String) {
        (
            "user",
            format!("<tool_response>\n{content}\n</tool_response>"),
        )
    }

    fn open(&self) -> &str {
        Self::OPEN
    }

    fn close(&self) -> &str {
        Self::CLOSE
    }

    fn grammar(&self, tools: &[ToolDefinition]) -> Result<String, LmError> {
        // Built as text, not through `serde_json::json!`: `serde_json` orders
        // object keys alphabetically, which would put `arguments` before `name`
        // and force the model to fill in arguments before committing to a tool.
        let branches = tools
            .iter()
            .map(|tool| {
                format!(
                    "{{\"type\":\"object\",\"properties\":{{\"name\":{{\"const\":{}}},\
                     \"arguments\":{}}},\"required\":[\"name\",\"arguments\"],\
                     \"additionalProperties\":false}}",
                    quoted(&tool.name),
                    tool.parameters,
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let body = llama_cpp_2::json_schema_to_grammar(&format!("{{\"oneOf\":[{branches}]}}"))
            .map_err(|error| LmError::Engine(format!("tool-call grammar: {error}")))?;
        // `root` and `space` come from the converted schema; the entry rule adds
        // the delimiters around it.
        Ok(format!(
            "{body}\n{TOOL_CALL_ROOT} ::= \"{}\" space root \"{}\"\n",
            Self::OPEN,
            Self::CLOSE,
        ))
    }

    fn parse(&self, body: &str) -> Result<(Arc<str>, serde_json::Value), LmError> {
        let call: serde_json::Value = serde_json::from_str(body.trim())
            .map_err(|error| LmError::InvalidToolArguments(format!("tool-call body: {error}")))?;
        let name = call
            .get("name")
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or(LmError::IncompleteToolCall)?;
        let arguments = call.get("arguments").ok_or(LmError::IncompleteToolCall)?;
        if !arguments.is_object() {
            return Err(LmError::ToolArgumentsNotObject);
        }
        Ok((Arc::from(name), arguments.clone()))
    }
}

/// `text` as a JSON string literal.
fn quoted(text: &str) -> serde_json::Value {
    serde_json::Value::String(text.to_owned())
}

/// Escape ECMAScript metacharacters so a delimiter is matched literally by
/// llama.cpp's lazy-grammar trigger patterns.
pub(crate) fn regex_escape(literal: &str) -> String {
    let mut escaped = String::with_capacity(literal.len());
    for character in literal.chars() {
        if "^$\\.*+?()[]{}|/".contains(character) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str, parameters: serde_json::Value) -> ToolDefinition {
        ToolDefinition::new(name, format!("{name} does a thing"), parameters).expect("valid tool")
    }

    fn dispatch_tools() -> Vec<ToolDefinition> {
        vec![
            tool(
                "dispatch_task",
                json!({
                    "type": "object",
                    "properties": {
                        "task": { "type": "string" },
                        "context": { "type": ["string", "null"] },
                    },
                    "required": ["task"],
                }),
            ),
            tool(
                "update_task",
                json!({
                    "type": "object",
                    "properties": {
                        "task_id": { "type": "string" },
                        "message": { "type": "string" },
                    },
                    "required": ["task_id", "message"],
                }),
            ),
        ]
    }

    #[test]
    fn declarations_carry_every_tool_in_a_tools_block() {
        let block = ChatMlXml.declarations(&dispatch_tools());
        assert!(block.contains("<tools>") && block.contains("</tools>"));
        assert!(block.contains(r#""name": "dispatch_task""#));
        assert!(block.contains(r#""name": "update_task""#));
        assert!(
            block.contains(r#"{"name": <function-name>, "arguments": <args-json-object>}"#),
            "the call shape Qwen's own template states must be present: {block}"
        );
    }

    #[test]
    fn declarations_and_grammar_are_byte_stable_across_calls() {
        // The declaration block is a prompt prefix, so a byte that moves between
        // turns costs a full KV-cache re-prefill.
        let tools = dispatch_tools();
        assert_eq!(
            ChatMlXml.declarations(&tools),
            ChatMlXml.declarations(&tools)
        );
        assert_eq!(
            ChatMlXml.grammar(&tools).expect("grammar"),
            ChatMlXml.grammar(&tools).expect("grammar")
        );
    }

    #[test]
    fn grammar_defines_the_entry_rule_and_names_every_tool() {
        let grammar = ChatMlXml.grammar(&dispatch_tools()).expect("grammar");
        assert!(grammar.contains(&format!("{TOOL_CALL_ROOT} ::=")));
        assert!(grammar.contains(r#""<tool_call>""#) && grammar.contains(r#""</tool_call>""#));
        assert!(grammar.contains(r#""\"dispatch_task\"""#), "{grammar}");
        assert!(grammar.contains(r#""\"update_task\"""#), "{grammar}");
    }

    #[test]
    fn grammar_puts_the_name_before_the_arguments() {
        // `const` and `oneOf` both convert; alphabetical key order would make the
        // model choose arguments before it has committed to a tool.
        let grammar = ChatMlXml.grammar(&dispatch_tools()).expect("grammar");
        let alternative = grammar
            .lines()
            .find(|line| line.starts_with("alternative-0 ::="))
            .expect("a branch per tool");
        let name = alternative.find("name-kv").expect("name in the branch");
        let arguments = alternative.find("arguments-kv").expect("arguments");
        assert!(name < arguments, "{alternative}");
    }

    #[test]
    fn grammar_for_one_tool_still_converts() {
        let grammar = ChatMlXml
            .grammar(&[tool("only", json!({ "type": "object" }))])
            .expect("grammar");
        assert!(grammar.contains(&format!("{TOOL_CALL_ROOT} ::=")));
        assert!(grammar.contains("space ::="), "{grammar}");
    }

    #[test]
    fn assistant_tool_calls_round_trip_into_the_prompt() {
        let call = ToolCall {
            id: Arc::from("call-1"),
            name: Arc::from("dispatch_task"),
            arguments_json: Arc::from(r#"{"task":"research otters"}"#),
        };
        assert_eq!(
            ChatMlXml.tool_calls(std::slice::from_ref(&call)),
            "\n<tool_call>\n{\"name\": \"dispatch_task\", \"arguments\": \
             {\"task\":\"research otters\"}}\n</tool_call>"
        );
        // A parse of what we rendered must return what we started with.
        let body = ChatMlXml
            .tool_calls(&[call])
            .trim_start_matches("\n<tool_call>")
            .trim_end_matches("</tool_call>")
            .to_string();
        let (name, arguments) = ChatMlXml.parse(&body).expect("round trip");
        assert_eq!(&*name, "dispatch_task");
        assert_eq!(arguments, json!({ "task": "research otters" }));
    }

    #[test]
    fn tool_results_use_a_role_the_chat_template_knows() {
        // llama.cpp's built-in ChatML renderer passes the role through verbatim,
        // so a `tool` role would emit `<|im_start|>tool`, which Qwen never saw.
        let (role, content) = ChatMlXml.tool_result("task accepted");
        assert_eq!(role, "user");
        assert_eq!(content, "<tool_response>\ntask accepted\n</tool_response>");
    }

    #[test]
    fn parse_reads_a_well_formed_body() {
        let (name, arguments) = ChatMlXml
            .parse("\n{\"name\": \"dispatch_task\", \"arguments\": {\"task\": \"x\"}}\n")
            .expect("parse");
        assert_eq!(&*name, "dispatch_task");
        assert_eq!(arguments, json!({ "task": "x" }));
    }

    #[test]
    fn parse_rejects_a_truncated_body() {
        let error = ChatMlXml
            .parse(r#"{"name": "dispatch_task", "arguments": {"task": "resea"#)
            .expect_err("truncated body");
        assert!(matches!(error, LmError::InvalidToolArguments(_)));
    }

    #[test]
    fn parse_rejects_a_body_that_is_not_json() {
        assert!(matches!(
            ChatMlXml.parse("dispatch_task(task=x)"),
            Err(LmError::InvalidToolArguments(_))
        ));
    }

    #[test]
    fn parse_rejects_a_missing_or_empty_name() {
        assert_eq!(
            ChatMlXml.parse(r#"{"arguments": {}}"#),
            Err(LmError::IncompleteToolCall)
        );
        assert_eq!(
            ChatMlXml.parse(r#"{"name": "", "arguments": {}}"#),
            Err(LmError::IncompleteToolCall)
        );
    }

    #[test]
    fn parse_rejects_missing_or_non_object_arguments() {
        assert_eq!(
            ChatMlXml.parse(r#"{"name": "t"}"#),
            Err(LmError::IncompleteToolCall)
        );
        assert_eq!(
            ChatMlXml.parse(r#"{"name": "t", "arguments": "x"}"#),
            Err(LmError::ToolArgumentsNotObject)
        );
    }

    #[test]
    fn supported_models_match_real_gguf_names_case_insensitively() {
        // The names the two Qwen GGUFs in `models/` actually report.
        assert!(ChatMlXml.supports("qwen2.5-0.5b-instruct"));
        assert!(ChatMlXml.supports("Qwen3 1.7B"));
        assert!(ChatMlXml.supports("Nous Hermes 2 Pro"));
        assert!(!ChatMlXml.supports("gemma-4-e2b-it"));
        assert!(!ChatMlXml.supports("Meta Llama 3.1 8B Instruct"));
    }

    #[test]
    fn regex_escape_leaves_a_plain_delimiter_alone_and_escapes_metacharacters() {
        assert_eq!(regex_escape("<tool_call>"), "<tool_call>");
        assert_eq!(regex_escape("[TOOL_CALLS]"), r"\[TOOL_CALLS\]");
    }
}
