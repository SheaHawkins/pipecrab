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
/// without punctuation. This is a deliberately simple heuristic — it will split
/// after an abbreviation like "Dr." — kept honest for v1.
///
/// # State and barge-in
///
/// The only state is `emitted`: how many bytes of the current generation have
/// already been turned into sentences. Because agent partials are append-only,
/// that byte offset stays valid as the text grows; the terminal
/// [`Final`](Finality::Final) resets it for the next generation. A barge-in
/// [`Interrupt`](SystemFrame::Interrupt) also resets it, abandoning a
/// half-emitted generation. All mutation lives in the synchronous `decide_*`, so
/// nothing tears when an emit is dropped.
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
            DataFrame::Transcript(Transcript { role: Role::Agent, finality, text }) => {
                // A fresh, shorter generation after an un-finalized one (no
                // terminal Final, e.g. it was interrupted): restart the offset so
                // slicing stays in bounds.
                if self.emitted > text.len() {
                    self.emitted = 0;
                }
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
        Decision { disposition: Disposition::Drop, effects }
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
    async fn perform(&self, EmitSentence(text): EmitSentence, out: &Outbound) -> Result<(), StageError> {
        // Ignore the send error: it only happens once the sink has gone away
        // during shutdown, matching the runtime's own forward path.
        let _ = out.send_data(Transcript::agent_final(text).into()).await;
        Ok(())
    }
}

/// Byte length of the leading complete sentence in `s`, or `None` if `s` holds
/// no completed sentence yet.
///
/// A sentence ends at a `.`/`!`/`?` run (plus trailing closers like `"`/`'`/`)`)
/// that is *followed by whitespace* — the whitespace is what confirms the
/// boundary, so a partial that ends mid-token (or mid-number like `3.`) is not
/// split prematurely. The returned length spans up to, but not including, that
/// whitespace. All the matched bytes are ASCII, so the length lands on a char
/// boundary of the UTF-8 `s`.
fn leading_sentence(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        if matches!(bytes[i], b'.' | b'!' | b'?') {
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
