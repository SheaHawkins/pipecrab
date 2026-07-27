//! Splitting a decoded token stream into spoken text and tool-call bodies.

use std::sync::Arc;

use pipecrab_lm::LmError;

/// One piece pulled out of the token stream.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Intercepted {
    /// Text to speak.
    Text(String),
    /// A captured call body — everything between the delimiters.
    Call(String),
}

/// Separates tool calls from speech as tokens arrive.
///
/// Text flows straight through except for a trailing run that could still grow
/// into the open delimiter: emitting `<tool` and only then discovering it opened
/// a call means the fragment was already spoken.
pub(crate) struct CallInterceptor<'a> {
    open: &'a str,
    close: &'a str,
    buffer: String,
    capturing: bool,
}

impl<'a> CallInterceptor<'a> {
    /// Intercept calls delimited by `open` and `close`.
    pub(crate) fn new(open: &'a str, close: &'a str) -> Self {
        Self {
            open,
            close,
            buffer: String::new(),
            capturing: false,
        }
    }

    /// Feed one decoded token piece, appending whatever it completes to `out`.
    pub(crate) fn push(&mut self, piece: &str, out: &mut Vec<Intercepted>) {
        self.buffer.push_str(piece);
        loop {
            if self.capturing {
                let Some(end) = self.buffer.find(self.close) else {
                    return;
                };
                out.push(Intercepted::Call(self.buffer[..end].to_string()));
                self.buffer.drain(..end + self.close.len());
                self.capturing = false;
            } else if let Some(start) = self.buffer.find(self.open) {
                if start > 0 {
                    out.push(Intercepted::Text(self.buffer[..start].to_string()));
                }
                self.buffer.drain(..start + self.open.len());
                self.capturing = true;
            } else {
                let speakable = self.buffer.len() - self.withheld();
                if speakable > 0 {
                    out.push(Intercepted::Text(self.buffer.drain(..speakable).collect()));
                }
                return;
            }
        }
    }

    /// Close the stream, returning any text still withheld.
    ///
    /// Fails if generation stopped inside a call — under `max_tokens`, a body can
    /// be cut mid-JSON, and half a call is not a call.
    pub(crate) fn finish(&mut self) -> Result<Option<Intercepted>, LmError> {
        if self.capturing {
            return Err(LmError::IncompleteToolCall);
        }
        Ok((!self.buffer.is_empty()).then(|| Intercepted::Text(std::mem::take(&mut self.buffer))))
    }

    /// Length of the buffer's trailing run that is still a proper prefix of the
    /// open delimiter.
    fn withheld(&self) -> usize {
        self.open
            .char_indices()
            .skip(1)
            .map(|(length, _)| length)
            .filter(|&length| self.buffer.ends_with(&self.open[..length]))
            .max()
            .unwrap_or(0)
    }
}

/// Mint a globally unique tool-call id — local models emit none.
pub(crate) fn mint_call_id() -> Arc<str> {
    Arc::from(format!("call-{}", uuid::Uuid::new_v4()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    /// Drive the interceptor with a scripted token sequence.
    fn run(pieces: &[&str]) -> (Vec<Intercepted>, Result<Option<Intercepted>, LmError>) {
        let mut interceptor = CallInterceptor::new(OPEN, CLOSE);
        let mut out = Vec::new();
        for piece in pieces {
            interceptor.push(piece, &mut out);
        }
        let tail = interceptor.finish();
        (out, tail)
    }

    fn text(items: &[Intercepted]) -> String {
        items.iter().fold(String::new(), |mut all, item| {
            if let Intercepted::Text(piece) = item {
                all.push_str(piece);
            }
            all
        })
    }

    fn calls(items: &[Intercepted]) -> Vec<&str> {
        items
            .iter()
            .filter_map(|item| match item {
                Intercepted::Call(body) => Some(body.as_str()),
                Intercepted::Text(_) => None,
            })
            .collect()
    }

    #[test]
    fn plain_text_is_emitted_as_it_arrives() {
        // Speech must not be delayed: every piece leaves on the push that fed it.
        let mut interceptor = CallInterceptor::new(OPEN, CLOSE);
        let mut out = Vec::new();
        for piece in ["Sure", ", I", " can help."] {
            let before = out.len();
            interceptor.push(piece, &mut out);
            assert_eq!(out.len(), before + 1, "piece {piece:?} was held back");
        }
        assert_eq!(text(&out), "Sure, I can help.");
        assert!(interceptor.finish().expect("clean finish").is_none());
    }

    #[test]
    fn a_call_arriving_as_one_token_is_captured_whole() {
        let (out, tail) = run(&[OPEN, "\n{\"name\": \"t\", \"arguments\": {}}\n", CLOSE]);
        assert_eq!(calls(&out), ["\n{\"name\": \"t\", \"arguments\": {}}\n"]);
        assert_eq!(text(&out), "");
        assert!(tail.expect("clean finish").is_none());
    }

    #[test]
    fn a_trigger_split_across_tokens_never_reaches_the_transcript() {
        let (out, tail) = run(&["<", "tool", "_call", ">", "{}", "</tool", "_call>"]);
        assert_eq!(text(&out), "", "a fragment of the call syntax was spoken");
        assert_eq!(calls(&out), ["{}"]);
        assert!(tail.expect("clean finish").is_none());
    }

    #[test]
    fn withheld_text_that_is_not_a_trigger_is_flushed_intact() {
        let (mut out, tail) = run(&["a < b, ", "<too", "ls>", " done"]);
        out.extend(tail.expect("clean finish"));
        assert_eq!(text(&out), "a < b, <tools> done");
        assert!(calls(&out).is_empty());
    }

    #[test]
    fn a_dangling_prefix_at_the_end_is_still_spoken() {
        let (out, tail) = run(&["done <"]);
        assert_eq!(text(&out), "done ");
        assert_eq!(
            tail.expect("clean finish"),
            Some(Intercepted::Text("<".into()))
        );
    }

    #[test]
    fn text_before_and_after_a_call_both_survive() {
        let (mut out, tail) = run(&["One moment. ", OPEN, "{\"a\":1}", CLOSE, " Done."]);
        out.extend(tail.expect("clean finish"));
        assert_eq!(
            out,
            [
                Intercepted::Text("One moment. ".into()),
                Intercepted::Call("{\"a\":1}".into()),
                Intercepted::Text(" Done.".into()),
            ]
        );
    }

    #[test]
    fn an_unterminated_call_fails_rather_than_yielding_half_of_one() {
        let (out, tail) = run(&[OPEN, "{\"name\": \"dispatch_task\", \"argum"]);
        assert!(calls(&out).is_empty(), "a partial call escaped");
        assert_eq!(tail, Err(LmError::IncompleteToolCall));
    }

    #[test]
    fn two_sequential_calls_are_captured_separately() {
        // The grammar admits one call per turn; the interceptor does not depend
        // on that, so a dialect without a grammar still reads both.
        let (out, tail) = run(&[OPEN, "{\"a\":1}", CLOSE, OPEN, "{\"b\":2}", CLOSE]);
        assert_eq!(calls(&out), ["{\"a\":1}", "{\"b\":2}"]);
        assert!(tail.expect("clean finish").is_none());
    }

    #[test]
    fn minted_ids_are_unique() {
        // The id becomes a dispatch idempotency key: a repeat is silently
        // deduplicated by the gateway and the task never runs.
        let ids: std::collections::HashSet<_> = (0..1_000).map(|_| mint_call_id()).collect();
        assert_eq!(ids.len(), 1_000);
    }
}
