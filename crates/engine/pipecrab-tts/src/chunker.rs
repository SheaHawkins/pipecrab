//! [`SentenceChunker`]: splits a streaming agent generation into one final agent
//! [`Transcript`] per sentence, so [`TtsStage`](crate::TtsStage) can begin
//! speaking the first sentence before the model has produced the last.

use std::sync::Arc;

use async_trait::async_trait;
use pipecrab_core::{
    DataFrame, Decision, Direction, Disposition, Finality, Processor, Role, SystemFrame, Transcript,
};
use pipecrab_runtime::{Outbound, Stage, StageError};

/// Adapts a streaming agent generation into per-sentence final transcripts.
///
/// The language-model stage upstream emits an agent generation as a run of
/// append-only [`Partial`](Finality::Partial) transcripts (each the full text so
/// far) followed by one [`Final`](Finality::Final). Feeding that straight to
/// [`TtsStage`](crate::TtsStage) would delay all speech until the whole reply is
/// generated. This stage instead watches the growing text and, each time a
/// sentence completes, emits it as its own agent [`Final`](Finality::Final) — so
/// synthesis of sentence one starts while the model is still writing sentence
/// two. This is the "generation for [`Role::Agent`] may be a single sentence"
/// note on [`Finality::Final`].
///
/// # What it consumes and emits
///
/// It *consumes* the raw agent stream (dropping the partials and the terminal
/// final) and *emits* one [`EmitSentence`] effect per completed sentence; every
/// non-agent frame forwards untouched. A sentence completes at `.`, `!`, or `?`
/// (with trailing closers like `"`/`)`) followed by whitespace; the terminal
/// [`Final`](Finality::Final) flushes any trailing text as a last sentence even
/// without punctuation.
///
/// A period is *not* a boundary when the word before it is a known abbreviation
/// (`Mr`, `Mrs`, `Ms`, `Dr`, `Prof`, `St`, `vs`, …) or a lone initial (`J.`), so
/// "Dr. Smith" and "U.S." stay intact; `!` and `?` always end a sentence. The
/// tradeoff is deliberate under-splitting: an abbreviation that genuinely ends a
/// sentence (`… etc. Next`) is held until the next real boundary or the
/// generation's [`Final`](Finality::Final).
///
/// # State and barge-in
///
/// The only state is `emitted`: how many bytes of the current generation have
/// already been turned into sentences. Because agent partials are append-only,
/// that byte offset stays valid as the text grows; the terminal
/// [`Final`](Finality::Final) resets it for the next generation. A barge-in
/// [`Interrupt`](SystemFrame::Interrupt) also resets it, abandoning a
/// half-emitted generation. Those are the *only* resets: a new generation must
/// be preceded by one, so an offset that outruns the current text is corrupt
/// state — a generation abandoned without an [`Interrupt`](SystemFrame::Interrupt)
/// — and panics rather than silently recovering. All mutation lives in the
/// synchronous `decide_*`, so nothing tears when an emit is dropped.
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

/// One completed sentence to forward as a final agent transcript:
/// [`SentenceChunker`]'s [`Processor::Effect`]. Emitted by `decide_data`,
/// interpreted by `perform`. Its inner text is private — only the chunker
/// constructs one.
pub struct EmitSentence(Arc<str>);

impl SentenceChunker {
    /// Drain every complete sentence starting at byte `from` in `text`, pushing
    /// an [`EmitSentence`] for each, and return the new offset just past the last
    /// one emitted.
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

/// Known abbreviations that carry a trailing period without ending a sentence.
/// Lowercased; matched case-insensitively against the word before a `.`. Multi-dot
/// forms (`e.g.`, `U.S.`) need no entry — their pieces are lone initials, which
/// [`ends_with_abbreviation`] already suppresses.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "vol", "fig", "gen",
    "sen", "rep", "gov", "col", "capt", "lt", "sgt", "rev", "hon",
];

/// Byte length of the leading complete sentence in `s`, or `None` if `s` holds
/// no completed sentence yet.
///
/// A sentence ends at a `.`/`!`/`?` run (plus trailing closers like `"`/`'`/`)`)
/// that is *followed by whitespace* — the whitespace is what confirms the
/// boundary, so a partial that ends mid-token (or mid-number like `3.`) is not
/// split prematurely. A `.` is skipped when the word before it is a known
/// abbreviation or a lone initial (see [`ends_with_abbreviation`]); `!` and `?`
/// always terminate. The returned length spans up to, but not including, that
/// whitespace. All the matched bytes are ASCII, so the length lands on a char
/// boundary of the UTF-8 `s`.
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

/// True if `before` (the text up to, but not including, a `.`) ends with a word
/// that keeps the period from closing a sentence: a known [`ABBREVIATIONS`]
/// entry, or a single-letter initial like `J`. The word is the trailing run of
/// ASCII letters and apostrophes, so contractions (`don't.`) read as ordinary
/// words and lone acronym letters (`U.S.`) read as initials.
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
