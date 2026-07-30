//! Decoding a completion into [`ModelDelta`]s: an incremental SSE reader and the
//! accumulator that reassembles fragmented `tool_calls` into complete calls.
//!
//! [`SseDecoder`] is a pure state machine over response bytes — it never touches
//! reqwest — so the protocol is tested without a server.

use std::collections::VecDeque;
use std::sync::Arc;

use futures::stream::{self, StreamExt};
use pipecrab_lm::{LmError, ModelDelta, ModelStream};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::model::Epoch;
use crate::wire::{ChatChunk, ChatResponse, ToolCallFragment, error_detail};

/// The sentinel event that ends an OpenAI stream.
const DONE: &str = "[DONE]";

/// One tool call being assembled from the fragments a host streams.
struct OpenCall {
    /// The call's position in the turn; the accumulator's key.
    index: u32,
    /// The provider's call id, if it sent one.
    id: Option<String>,
    /// The function name; the first non-empty fragment wins.
    name: String,
    /// Accumulated `arguments` JSON text.
    arguments: String,
}

impl OpenCall {
    /// An empty call at `index`.
    fn new(index: u32) -> Self {
        Self {
            index,
            id: None,
            name: String::new(),
            arguments: String::new(),
        }
    }

    /// A call that arrived whole — the non-streaming path.
    fn complete(fragment: ToolCallFragment) -> Self {
        let (name, arguments) = match fragment.function {
            Some(function) => (
                function.name.unwrap_or_default(),
                function.arguments.unwrap_or_default(),
            ),
            None => (String::new(), String::new()),
        };
        Self {
            index: fragment.index.unwrap_or(0),
            id: fragment.id,
            name,
            arguments,
        }
    }

    /// Whether the arguments seen so far are a finished JSON object. A balanced
    /// object can never be a prefix of a longer one — its only closing brace is
    /// its last character — so this is a sound completeness test. Empty
    /// arguments are excluded: for a tool that takes some, they mean "not yet".
    fn is_complete(&self) -> bool {
        let arguments = self.arguments.trim();
        !arguments.is_empty() && parse_arguments(arguments).is_ok()
    }
}

/// Turn one assembled call into a delta, minting an id if the host sent none.
fn flush(call: OpenCall) -> Result<ModelDelta, LmError> {
    let arguments = parse_arguments(&call.arguments)?;
    let id = call
        .id
        .filter(|id| !id.is_empty())
        .unwrap_or_else(mint_call_id);
    ModelDelta::tool_call(id, call.name, arguments)
}

/// Read accumulated argument text as a JSON value.
fn parse_arguments(text: &str) -> Result<Value, LmError> {
    let text = text.trim();
    // Zero-argument tools stream `""` on several hosts; failing them would be a
    // false negative.
    if text.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(text).map_err(|error| LmError::InvalidToolArguments(error.to_string()))
}

/// A call id for a host that streams a call without one. It only has to be
/// unique within the turn for the tool result to correlate.
fn mint_call_id() -> String {
    format!("call_{}", Uuid::new_v4())
}

/// An incremental reader for a Chat Completions SSE body: bytes in, deltas out.
#[derive(Default)]
pub(crate) struct SseDecoder {
    /// Bytes of a line not yet terminated by `\n`.
    line: Vec<u8>,
    /// The `data:` payloads of the event being assembled.
    event: String,
    /// Tool calls under assembly, kept in index order.
    calls: Vec<OpenCall>,
    /// `[DONE]` seen — nothing further will arrive.
    done: bool,
}

impl SseDecoder {
    /// Feed the next slice of response body, returning whatever it completed.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<Result<ModelDelta, LmError>> {
        let mut out = Vec::new();
        for &byte in bytes {
            if byte == b'\n' {
                // A newline never appears inside a UTF-8 multibyte sequence, so
                // splitting here cannot cut a character in half.
                let line = std::mem::take(&mut self.line);
                self.line_ready(&line, &mut out);
            } else {
                self.line.push(byte);
            }
        }
        out
    }

    /// End of body: dispatch a trailing unterminated event and flush every call
    /// still open. A host that ends without `[DONE]` or a `finish_reason` still
    /// gets its tool calls.
    pub(crate) fn finish(&mut self) -> Vec<Result<ModelDelta, LmError>> {
        let mut out = Vec::new();
        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            self.line_ready(&line, &mut out);
        }
        self.dispatch(&mut out);
        self.flush_all(&mut out);
        out
    }

    /// Whether the stream announced its own end.
    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    /// Handle one complete line of the body.
    fn line_ready(&mut self, line: &[u8], out: &mut Vec<Result<ModelDelta, LmError>>) {
        let line = String::from_utf8_lossy(strip_cr(line));
        if line.is_empty() {
            self.dispatch(out);
            return;
        }
        if line.starts_with(':') {
            // A comment, which is how hosts keep the connection alive.
            return;
        }
        let Some(data) = line.strip_prefix("data:") else {
            // `event:`, `id:`, `retry:`, or a field this adapter does not act on.
            return;
        };
        // Per SSE, consecutive data lines of one event join with a newline.
        if !self.event.is_empty() {
            self.event.push('\n');
        }
        self.event.push_str(data.strip_prefix(' ').unwrap_or(data));
    }

    /// Decode the assembled event.
    fn dispatch(&mut self, out: &mut Vec<Result<ModelDelta, LmError>>) {
        let event = std::mem::take(&mut self.event);
        let event = event.trim();
        if event.is_empty() {
            return;
        }
        if event == DONE {
            self.done = true;
            self.flush_all(out);
            return;
        }
        let chunk: ChatChunk = match serde_json::from_str(event) {
            Ok(chunk) => chunk,
            Err(error) => {
                out.push(Err(LmError::ProviderStream(format!(
                    "undecodable completion chunk: {error}"
                ))));
                return;
            }
        };
        if let Some(error) = &chunk.error {
            out.push(Err(LmError::ProviderStream(error_detail(error))));
            return;
        }
        // Only the first choice: `n > 1` is out of scope, and a usage-only
        // trailer carries no choices at all.
        let Some(choice) = chunk.choices.first() else {
            return;
        };
        if let Some(content) = choice.delta.content.as_deref().filter(|c| !c.is_empty()) {
            out.push(Ok(ModelDelta::Text(Arc::from(content))));
        }
        for fragment in &choice.delta.tool_calls {
            self.accumulate(fragment, out);
        }
        if choice.finish_reason.is_some() {
            self.flush_all(out);
        }
    }

    /// Merge one fragment into the call it belongs to.
    fn accumulate(
        &mut self,
        fragment: &ToolCallFragment,
        out: &mut Vec<Result<ModelDelta, LmError>>,
    ) {
        let index = fragment.index.unwrap_or(0);
        // A finished lower-index call need not wait for the rest of a parallel
        // turn. Guarded on completeness, so a host that interleaves indices
        // cannot have a half-streamed call emitted out from under it.
        self.flush_complete_below(index, out);

        let id = fragment.id.as_deref().filter(|id| !id.is_empty());
        let position = match self.calls.iter().position(|call| call.index == index) {
            // A different id at the same index means a new call: a host that
            // omits `index` altogether still gets one call per id.
            Some(position) if supersedes(&self.calls[position], id) => {
                let call = self.calls.remove(position);
                out.push(flush(call));
                self.open(index)
            }
            Some(position) => position,
            None => self.open(index),
        };

        let call = &mut self.calls[position];
        if call.id.is_none() {
            call.id = id.map(str::to_owned);
        }
        if let Some(function) = &fragment.function {
            if let Some(name) = function.name.as_deref().filter(|name| !name.is_empty())
                && call.name.is_empty()
            {
                call.name.push_str(name);
            }
            if let Some(arguments) = &function.arguments {
                call.arguments.push_str(arguments);
            }
        }
    }

    /// Start a call at `index`, keeping the list ordered; returns its position.
    fn open(&mut self, index: u32) -> usize {
        let position = self
            .calls
            .iter()
            .position(|call| call.index > index)
            .unwrap_or(self.calls.len());
        self.calls.insert(position, OpenCall::new(index));
        position
    }

    /// Emit every open call below `index` whose arguments are already complete.
    fn flush_complete_below(&mut self, index: u32, out: &mut Vec<Result<ModelDelta, LmError>>) {
        let mut position = 0;
        while position < self.calls.len() {
            if self.calls[position].index < index && self.calls[position].is_complete() {
                let call = self.calls.remove(position);
                out.push(flush(call));
            } else {
                position += 1;
            }
        }
    }

    /// Emit every open call, in index order, complete or not — an incomplete one
    /// surfaces as an error rather than reaching the pipeline half-formed.
    fn flush_all(&mut self, out: &mut Vec<Result<ModelDelta, LmError>>) {
        out.extend(self.calls.drain(..).map(flush));
    }
}

/// Whether a fragment's id belongs to a different call than the one open here.
fn supersedes(call: &OpenCall, id: Option<&str>) -> bool {
    matches!((call.id.as_deref(), id), (Some(open), Some(new)) if open != new)
}

/// Drop one trailing `\r`, so CRLF framing reads the same as LF.
fn strip_cr(line: &[u8]) -> &[u8] {
    match line {
        [head @ .., b'\r'] => head,
        _ => line,
    }
}

/// Decode a whole non-streamed completion body.
pub(crate) fn decode_response(bytes: &[u8]) -> Vec<Result<ModelDelta, LmError>> {
    let response: ChatResponse = match serde_json::from_slice(bytes) {
        Ok(response) => response,
        Err(error) => {
            return vec![Err(LmError::ProviderStream(format!(
                "undecodable completion: {error}"
            )))];
        }
    };
    if let Some(error) = &response.error {
        return vec![Err(LmError::ProviderStream(error_detail(error)))];
    }
    let Some(choice) = response.choices.into_iter().next() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(content) = choice.message.content.as_deref().filter(|c| !c.is_empty()) {
        out.push(Ok(ModelDelta::Text(Arc::from(content))));
    }
    out.extend(
        choice
            .message
            .tool_calls
            .into_iter()
            .map(|fragment| flush(OpenCall::complete(fragment))),
    );
    out
}

/// The state one streamed generation carries between polls.
struct SseState {
    response: reqwest::Response,
    decoder: SseDecoder,
    queue: VecDeque<Result<ModelDelta, LmError>>,
    epoch: Epoch,
    /// No further body reads are wanted: `[DONE]`, end of body, or a failure.
    finished: bool,
}

/// Stream a response body as deltas, ending as soon as `epoch` moves.
pub(crate) fn sse_stream(response: reqwest::Response, epoch: Epoch) -> ModelStream {
    let state = SseState {
        response,
        decoder: SseDecoder::default(),
        queue: VecDeque::new(),
        epoch,
        finished: false,
    };
    stream::unfold(state, |mut state| async move {
        loop {
            // Checked before every item: a cancel between chunks, or one racing
            // a decoded-but-unseen delta, ends the turn here. Returning drops
            // the response, which closes the connection.
            if !state.epoch.is_current() {
                return None;
            }
            if let Some(item) = state.queue.pop_front() {
                if item.is_err() {
                    // A failed turn is over; nothing decoded after it is
                    // meaningful.
                    state.queue.clear();
                    state.finished = true;
                }
                return Some((item, state));
            }
            if state.finished {
                return None;
            }
            match state.response.chunk().await {
                Ok(Some(chunk)) => {
                    state.queue.extend(state.decoder.push(&chunk));
                    state.finished = state.decoder.is_done();
                }
                Ok(None) => {
                    state.finished = true;
                    state.queue.extend(state.decoder.finish());
                }
                Err(error) => {
                    state.finished = true;
                    state.queue.push_back(Err(LmError::ProviderStream(format!(
                        "completion stream failed: {error}"
                    ))));
                }
            }
        }
    })
    .boxed()
}

/// Stream already-decoded deltas — the non-streaming path — under the same
/// cancellation epoch.
pub(crate) fn decoded_stream(items: Vec<Result<ModelDelta, LmError>>, epoch: Epoch) -> ModelStream {
    stream::iter(items)
        .take_while(move |_| futures::future::ready(epoch.is_current()))
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a whole body to a fresh decoder and collect everything it produced.
    fn decode(body: &str) -> Vec<Result<ModelDelta, LmError>> {
        let mut decoder = SseDecoder::default();
        let mut out = decoder.push(body.as_bytes());
        out.extend(decoder.finish());
        out
    }

    /// Feed a body one byte at a time — every split point exercised at once.
    fn decode_byte_by_byte(body: &str) -> Vec<Result<ModelDelta, LmError>> {
        let mut decoder = SseDecoder::default();
        let mut out = Vec::new();
        for byte in body.as_bytes() {
            out.extend(decoder.push(&[*byte]));
        }
        out.extend(decoder.finish());
        out
    }

    fn text(items: &[Result<ModelDelta, LmError>]) -> String {
        items
            .iter()
            .filter_map(|item| match item {
                Ok(ModelDelta::Text(text)) => Some(text.to_string()),
                _ => None,
            })
            .collect()
    }

    fn calls(items: Vec<Result<ModelDelta, LmError>>) -> Vec<(String, String, String)> {
        items
            .into_iter()
            .filter_map(|item| match item {
                Ok(ModelDelta::ToolCall(call)) => Some((
                    call.id.to_string(),
                    call.name.to_string(),
                    call.arguments_json.to_string(),
                )),
                _ => None,
            })
            .collect()
    }

    fn errors(items: Vec<Result<ModelDelta, LmError>>) -> Vec<LmError> {
        items.into_iter().filter_map(Result::err).collect()
    }

    const TEXT_BODY: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    #[test]
    fn text_deltas_concatenate() {
        let items = decode(TEXT_BODY);
        assert_eq!(text(&items), "Hello");
        assert!(errors(items).is_empty());
    }

    #[test]
    fn a_body_split_at_every_byte_decodes_identically() {
        assert_eq!(text(&decode_byte_by_byte(TEXT_BODY)), "Hello");
    }

    #[test]
    fn crlf_comments_and_other_fields_are_tolerated() {
        let body = concat!(
            ": ping\r\n",
            "event: message\r\n",
            "id: 1\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\r\n",
            "\r\n",
            "data: [DONE]\r\n\r\n",
        );
        let items = decode(body);
        assert_eq!(text(&items), "hi");
        assert!(errors(items).is_empty());
    }

    #[test]
    fn multi_line_data_joins_with_a_newline() {
        // The two data lines are halves of one JSON document.
        let body = "data: {\"choices\":[{\"delta\":\ndata: {\"content\":\"hi\"}}]}\n\n";
        assert_eq!(text(&decode(body)), "hi");
    }

    #[test]
    fn empty_and_absent_content_emit_nothing() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":null}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{}}]}\n\n",
            "data: {\"choices\":[]}\n\n",
        );
        assert!(decode(body).is_empty());
    }

    #[test]
    fn reasoning_fields_never_reach_the_stream() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"still thinking\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\n",
        );
        let items = decode(body);
        assert_eq!(text(&items), "done");
        assert!(errors(items).is_empty());
    }

    #[test]
    fn done_ends_the_stream_and_flushes_an_open_call() {
        // `is_done` is what stops the stream pulling the body, so anything a
        // host writes after `[DONE]` is never read.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"ping\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let mut decoder = SseDecoder::default();
        let items = decoder.push(body.as_bytes());
        assert!(decoder.is_done());
        assert_eq!(calls(items).len(), 1);
    }

    #[test]
    fn one_tool_call_assembles_from_fragmented_arguments() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"weather\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"paris\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        assert_eq!(
            calls(decode(body)),
            vec![(
                "call_a".into(),
                "weather".into(),
                r#"{"city":"paris"}"#.into()
            )]
        );
    }

    #[test]
    fn a_call_arriving_whole_in_one_chunk_works() {
        let body = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"ping\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n";
        assert_eq!(
            calls(decode(body)),
            vec![("call_a".into(), "ping".into(), "{}".into())]
        );
    }

    #[test]
    fn two_calls_emit_in_index_order_and_the_first_does_not_wait() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"one\",\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"two\",\"arguments\":\"{\\\"b\\\":2}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        );
        let names: Vec<String> = calls(decode(body))
            .into_iter()
            .map(|(_, name, _)| name)
            .collect();
        assert_eq!(names, vec!["one".to_string(), "two".to_string()]);
    }

    #[test]
    fn an_interleaved_call_is_not_truncated_by_the_early_flush() {
        // Index 0 opens, index 1 arrives, then index 0's arguments finish. The
        // early flush must not have emitted a half-built call 0.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"one\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"two\",\"arguments\":\"{\\\"b\\\":2}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        );
        let decoded = calls(decode(body));
        assert_eq!(decoded.len(), 2, "{decoded:?}");
        assert!(
            decoded
                .iter()
                .any(|(_, name, arguments)| name == "one" && arguments == r#"{"a":1}"#),
            "{decoded:?}"
        );
    }

    #[test]
    fn a_new_id_at_the_same_index_starts_a_new_call() {
        // A host that omits `index` entirely still gets one call per id.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_a\",\"function\":{\"name\":\"one\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_b\",\"function\":{\"name\":\"two\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let ids: Vec<String> = calls(decode(body))
            .into_iter()
            .map(|(id, _, _)| id)
            .collect();
        assert_eq!(ids, vec!["call_a".to_string(), "call_b".to_string()]);
    }

    #[test]
    fn a_missing_id_is_minted() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"ping\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let decoded = calls(decode(body));
        assert_eq!(decoded.len(), 1);
        assert!(decoded[0].0.starts_with("call_"), "{decoded:?}");
        assert!(decoded[0].0.len() > "call_".len(), "{decoded:?}");
    }

    #[test]
    fn a_missing_name_is_an_error() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        assert!(
            matches!(
                errors(decode(body)).as_slice(),
                [LmError::IncompleteToolCall]
            ),
            "a nameless call cannot be dispatched"
        );
    }

    #[test]
    fn empty_arguments_become_an_empty_object() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        assert_eq!(
            calls(decode(body)),
            vec![("call_a".into(), "ping".into(), "{}".into())]
        );
    }

    #[test]
    fn unparsable_arguments_are_an_error() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"ping\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        assert!(
            matches!(
                errors(decode(body)).as_slice(),
                [LmError::InvalidToolArguments(_)]
            ),
            "half a call must not reach the pipeline"
        );
    }

    #[test]
    fn non_object_arguments_are_rejected() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"ping\",\"arguments\":\"[1,2]\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        assert!(
            matches!(
                errors(decode(body)).as_slice(),
                [LmError::ToolArgumentsNotObject]
            ),
            "arguments must be an object"
        );
    }

    #[test]
    fn end_of_body_without_done_still_flushes_an_open_call() {
        let body = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"ping\",\"arguments\":\"{}\"}}]}}]}\n\n";
        assert_eq!(calls(decode(body)).len(), 1);
    }

    #[test]
    fn an_error_chunk_becomes_a_provider_stream_error() {
        let body = "data: {\"error\":{\"message\":\"upstream is down\",\"code\":502}}\n\n";
        let reported = errors(decode(body));
        assert!(
            matches!(reported.as_slice(), [LmError::ProviderStream(message)] if message.as_str() == "upstream is down"),
            "{reported:?}"
        );
    }

    #[test]
    fn a_body_that_ends_mid_event_is_a_provider_stream_error() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"";
        assert!(
            matches!(
                errors(decode(body)).as_slice(),
                [LmError::ProviderStream(_)]
            ),
            "a truncated event is a failed stream"
        );
    }

    #[test]
    fn a_blocking_body_decodes_text_and_calls() {
        let body = r#"{
            "choices": [{
                "message": {
                    "content": "one moment",
                    "tool_calls": [
                        {"id": "call_a", "type": "function",
                         "function": {"name": "weather", "arguments": "{\"city\":\"paris\"}"}}
                    ]
                }
            }]
        }"#;
        let items = decode_response(body.as_bytes());
        assert_eq!(text(&items), "one moment");
        assert_eq!(
            calls(items),
            vec![(
                "call_a".into(),
                "weather".into(),
                r#"{"city":"paris"}"#.into()
            )]
        );
    }

    #[test]
    fn a_blocking_error_body_is_a_provider_stream_error() {
        let body = r#"{"error": "quota exceeded"}"#;
        let reported = errors(decode_response(body.as_bytes()));
        assert!(
            matches!(reported.as_slice(), [LmError::ProviderStream(message)] if message.as_str() == "quota exceeded"),
            "{reported:?}"
        );
    }
}
