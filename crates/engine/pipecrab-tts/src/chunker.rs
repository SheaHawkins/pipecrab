//! Splits agent output into sentences for low-latency synthesis.

use std::sync::Arc;

use async_trait::async_trait;
use pipecrab_core::{
    DataFrame, Decision, Direction, Disposition, Finality, Processor, Role, SystemFrame, Transcript,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

/// Converts a streaming agent generation into final transcripts per sentence.
///
/// It consumes append-only agent partials and emits each completed sentence as
/// [`Finality::Final`], allowing [`TtsStage`](crate::TtsStage) to start before
/// the full response is available.
///
/// # What it consumes and emits
///
/// A sentence boundary is `.`, `!`, or `?`, optional closing punctuation, then
/// whitespace. The generation's final frame flushes remaining text. Other
/// frames pass through.
///
/// Known abbreviations and lone initials do not end a sentence. This favors
/// under-splitting ambiguous periods until a later boundary or final frame.
///
/// # State and barge-in
///
/// The stage tracks the byte offset already emitted. Final frames and
/// [`SystemFrame::Interrupt`] reset it. An offset beyond the current append-only
/// text violates the input contract and panics.
pub struct SentenceChunker {
    /// Bytes of the in-flight generation already emitted as sentences.
    emitted: usize,
}

impl SentenceChunker {
    /// A chunker with no generation in flight.
    pub fn new() -> Self {
        Self { emitted: 0 }
    }
}

impl Default for SentenceChunker {
    fn default() -> Self {
        Self::new()
    }
}

/// A completed sentence to emit as a final agent transcript.
pub struct EmitSentence(Arc<str>);

impl SentenceChunker {
    /// Drains complete sentences and returns the next unconsumed byte offset.
    fn drain_sentences(from: usize, text: &str, effects: &mut Vec<EmitSentence>) -> usize {
        let mut cut = from;
        while let Some(len) = leading_sentence(&text[cut..]) {
            let sentence = text[cut..cut + len].trim();
            if !sentence.is_empty() {
                effects.push(EmitSentence(sentence.into()));
            }
            cut += len;
        }
        cut
    }
}

impl Processor for SentenceChunker {
    type Effect = EmitSentence;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<EmitSentence> {
        let (finality, text) = match frame {
            DataFrame::Transcript(Transcript {
                role: Role::Agent,
                finality,
                text,
            }) => {
                // Agent partials are append-only, so within a generation the
                // offset never outruns the text. If it does, a new generation
                // arrived without the Final/Interrupt that resets us — corrupt
                // state we refuse to paper over.
                assert!(
                    self.emitted <= text.len(),
                    "SentenceChunker offset {} outruns agent text of {} bytes: a generation \
                     was abandoned without an Interrupt",
                    self.emitted,
                    text.len(),
                );
                (*finality, text)
            }
            // User speech, audio, custom frames: not ours to chunk.
            _ => return Decision::forward(),
        };

        let mut effects = Vec::new();
        let cut = Self::drain_sentences(self.emitted, text, &mut effects);
        match finality {
            Finality::Partial { .. } => {
                // More text may still arrive; remember how far we have voiced.
                self.emitted = cut;
            }
            Finality::Final => {
                // End of generation: flush the trailing remainder as the last
                // sentence even if it has no terminator, then reset.
                let tail = text[cut..].trim();
                if !tail.is_empty() {
                    effects.push(EmitSentence(tail.into()));
                }
                self.emitted = 0;
            }
        }
        // Consume the raw agent frame; the per-sentence finals replace it.
        Decision {
            disposition: Disposition::Drop,
            effects,
        }
    }

    fn decide_system(&mut self, _dir: Direction, frame: &SystemFrame) -> Decision<EmitSentence> {
        // Barge-in abandons the in-flight generation; the next one starts clean.
        if matches!(frame, SystemFrame::Interrupt) {
            self.emitted = 0;
        }
        Decision::forward()
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Stage for SentenceChunker {
    async fn perform(
        &self,
        EmitSentence(text): EmitSentence,
        out: &Outbound,
    ) -> Result<(), StageError> {
        // Ignore the send error: it only happens once the sink has gone away
        // during shutdown, matching the runtime's own forward path.
        let _ = out.send_data(Transcript::agent_final(text).into()).await;
        Ok(())
    }
}

/// Lowercase abbreviations whose period does not end a sentence.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "vol", "fig", "gen",
    "sen", "rep", "gov", "col", "capt", "lt", "sgt", "rev", "hon",
];

/// Returns the byte length of the first complete sentence.
///
/// The boundary requires trailing whitespace, excludes that whitespace, and
/// ignores periods after abbreviations or lone initials.
fn leading_sentence(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        if matches!(bytes[i], b'.' | b'!' | b'?') {
            // A period after an abbreviation or a lone initial is not a boundary.
            if bytes[i] == b'.' && ends_with_abbreviation(&s[..i]) {
                continue;
            }
            let mut j = i + 1;
            while j < bytes.len()
                && matches!(bytes[j], b'.' | b'!' | b'?' | b'"' | b'\'' | b')' | b']')
            {
                j += 1;
            }
            if j < bytes.len() && bytes[j].is_ascii_whitespace() {
                return Some(j);
            }
        }
    }
    None
}

/// Returns whether a trailing word or initial prevents a period boundary.
fn ends_with_abbreviation(before: &str) -> bool {
    let bytes = before.as_bytes();
    let mut start = bytes.len();
    while start > 0 && (bytes[start - 1].is_ascii_alphabetic() || bytes[start - 1] == b'\'') {
        start -= 1;
    }
    let word = &before[start..];
    // A lone initial, e.g. "J" in "J. R. R.".
    if word.len() == 1 && word.as_bytes()[0].is_ascii_alphabetic() {
        return true;
    }
    ABBREVIATIONS.contains(&word.to_ascii_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A completed sentence's byte length is measured up to the confirming
    /// whitespace, so `leading_sentence` returns the index of that whitespace.
    #[test]
    fn leading_sentence_splits_on_terminator_then_whitespace() {
        assert_eq!(leading_sentence("one. two"), Some(4));
        assert_eq!(leading_sentence("stop! go"), Some(5));
        assert_eq!(leading_sentence("what? now"), Some(5));
        // Trailing closers are pulled into the sentence before the whitespace.
        assert_eq!(leading_sentence("she said \"hi.\" then"), Some(14));
    }

    #[test]
    fn leading_sentence_needs_the_confirming_whitespace() {
        // A terminator at the very end could still be mid-token in the next
        // partial, so it is not yet a boundary.
        assert_eq!(leading_sentence("done."), None);
        assert_eq!(leading_sentence("no terminator here"), None);
        // A period mid-number is not followed by whitespace, so never a boundary.
        assert_eq!(leading_sentence("pi is 3.14 today"), None);
    }

    #[test]
    fn leading_sentence_skips_abbreviations_and_initials() {
        // Titles and other known abbreviations do not end the sentence...
        assert_eq!(leading_sentence("Dr. Smith arrived"), None);
        assert_eq!(leading_sentence("see vol. 2 now"), None);
        // ...nor do lone initials, including the pieces of an acronym.
        assert_eq!(leading_sentence("J. R. R. Tolkien wrote"), None);
        assert_eq!(leading_sentence("the U.S. economy grew"), None);
        // But a real sentence *after* an abbreviation still splits.
        assert_eq!(leading_sentence("Dr. Smith left. Then"), Some(15));
        // A contraction is an ordinary word, not a suppressed one-letter token.
        assert_eq!(leading_sentence("I don't. Really"), Some("I don't.".len()));
        // `!`/`?` are never suppressed, abbreviation or not.
        assert_eq!(leading_sentence("Wait, Dr! Stop"), Some(9));
    }

    #[test]
    fn ends_with_abbreviation_classifies_the_trailing_word() {
        assert!(ends_with_abbreviation("Dr")); // known abbreviation
        assert!(ends_with_abbreviation("hello Mrs")); // case-insensitive, trailing word only
        assert!(ends_with_abbreviation("J")); // lone initial
        assert!(!ends_with_abbreviation("hello")); // ordinary word
        assert!(!ends_with_abbreviation("don't")); // contraction, not a lone "t"
        assert!(!ends_with_abbreviation("")); // nothing before the period
    }

    #[test]
    fn drain_sentences_emits_each_complete_sentence_and_returns_the_offset() {
        let mut effects = Vec::new();
        let cut = SentenceChunker::drain_sentences(0, "one. two. three", &mut effects);
        let texts: Vec<&str> = effects.iter().map(|EmitSentence(t)| &**t).collect();
        assert_eq!(texts, vec!["one.", "two."]);
        // Offset stops just past "one. two. " — the unfinished "three" remains.
        assert_eq!(cut, "one. two. ".len() - 1);
        assert_eq!(&"one. two. three"[cut..], " three");
    }

    #[test]
    fn drain_sentences_starts_from_the_given_offset() {
        let mut effects = Vec::new();
        // Pretend "one. " was already emitted: start past it.
        let cut = SentenceChunker::drain_sentences(4, "one. two. rest", &mut effects);
        let texts: Vec<&str> = effects.iter().map(|EmitSentence(t)| &**t).collect();
        assert_eq!(texts, vec!["two."]);
        assert_eq!(&"one. two. rest"[cut..], " rest");
    }

    fn agent_partial(text: &str) -> DataFrame {
        Transcript::agent_partial(text).into()
    }

    #[test]
    #[should_panic(expected = "outruns agent text")]
    fn shorter_generation_without_reset_panics() {
        let mut chunker = SentenceChunker::new();
        // Advances the offset well past a later, shorter generation's length.
        let _ = chunker.decide_data(&agent_partial("one two three. four five"));
        // A new, shorter generation with no intervening Interrupt/Final: corrupt
        // state that must fail loudly rather than silently reset.
        let _ = chunker.decide_data(&agent_partial("hi"));
    }
}
