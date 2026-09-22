//! Splits a sentence into the clauses [`KokoroTts`](crate::KokoroTts) speaks
//! one inference run at a time, and trims the silence where they meet.

/// Words a clause must reach before its punctuation may end it. A shorter one
/// ("Well,") stays with the next, as does a shorter remainder with the last.
const MIN_WORDS: usize = 4;

/// Split `text` after each `,` `;` `:` `—` `–` that ends a word, once the clause
/// so far has [`MIN_WORDS`] words. Each clause is trimmed; blank text yields
/// none.
///
/// Sentence ends are not split here: that is the
/// [`SentenceChunker`](pipecrab_tts::SentenceChunker)'s job, and it already
/// knows which periods end an abbreviation instead.
pub(crate) fn clauses(text: &str) -> Vec<&str> {
    let mut ends = Vec::new();
    let mut start = 0;
    let mut count = 0;
    for (word_start, word_end) in words(text) {
        count += 1;
        if count >= MIN_WORDS && ends_clause(&text[word_start..word_end]) {
            ends.push(word_end);
            start = word_end;
            count = 0;
        }
    }
    match ends.last_mut() {
        // A short remainder rides with the clause before it.
        Some(last) if count < MIN_WORDS => *last = text.len(),
        _ if !text[start..].trim().is_empty() => ends.push(text.len()),
        _ => {}
    }

    let mut from = 0;
    ends.into_iter()
        .map(|end| {
            let clause = text[from..end].trim();
            from = end;
            clause
        })
        .collect()
}

/// Amplitude below which a sample counts as silence: -40 dBFS.
const QUIET: f32 = 0.01;

/// Silence kept before a clause that continues a sentence: 100 ms at 24 kHz.
const JOIN_LEAD: usize = 2_400;

/// Silence kept after a clause the sentence continues past: 150 ms at 24 kHz.
const JOIN_TAIL: usize = 3_600;

/// Cut a clause's edge silence to a comma's pause where it meets another
/// clause of the same sentence: the lead when `joined_before`, the tail when
/// `joined_after`.
///
/// Kokoro pads every run with about 0.3 s of silence before and 0.45 s after,
/// so two clauses played back to back would pause for 0.7 s at each join.
pub(crate) fn trim_joins(samples: &[f32], joined_before: bool, joined_after: bool) -> &[f32] {
    let loud = |sample: &f32| sample.abs() >= QUIET;
    let (Some(first), Some(last)) = (
        samples.iter().position(loud),
        samples.iter().rposition(loud),
    ) else {
        return samples;
    };
    let start = match joined_before {
        true => first.saturating_sub(JOIN_LEAD),
        false => 0,
    };
    let end = match joined_after {
        true => (last + 1 + JOIN_TAIL).min(samples.len()),
        false => samples.len(),
    };
    &samples[start..end]
}

/// Byte ranges of the whitespace-separated words in `text`.
fn words(text: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut start = None;
    text.char_indices()
        .chain(std::iter::once((text.len(), ' ')))
        .filter_map(move |(at, c)| match (c.is_whitespace(), start) {
            (true, Some(word_start)) => {
                start = None;
                Some((word_start, at))
            }
            (false, None) => {
                start = Some(at);
                None
            }
            _ => None,
        })
}

/// Whether `word` ends in clause punctuation, looking past closing quotes and
/// brackets. A comma inside a word ("1,000") is not one.
fn ends_clause(word: &str) -> bool {
    word.trim_end_matches(['"', '\'', '”', '’', ')', ']'])
        .ends_with([',', ';', ':', '—', '–'])
}

#[cfg(test)]
mod tests {
    use super::{JOIN_LEAD, JOIN_TAIL, clauses, trim_joins};

    #[test]
    fn splits_a_long_sentence_at_its_commas() {
        assert_eq!(
            clauses(
                "The lamps would be lit up, casting a warm glow on the streets, \
                 as the night air filled with the sounds of the city."
            ),
            [
                "The lamps would be lit up,",
                "casting a warm glow on the streets,",
                "as the night air filled with the sounds of the city.",
            ]
        );
    }

    #[test]
    fn keeps_a_sentence_without_clause_punctuation_whole() {
        assert_eq!(
            clauses("A crab is a crustacean with ten legs."),
            ["A crab is a crustacean with ten legs."]
        );
    }

    #[test]
    fn keeps_a_short_opening_with_what_follows() {
        assert_eq!(
            clauses("Well, I think that is right."),
            ["Well, I think that is right."]
        );
    }

    #[test]
    fn folds_a_short_remainder_into_the_last_clause() {
        assert_eq!(
            clauses("I bought apples and pears, too."),
            ["I bought apples and pears, too."]
        );
    }

    #[test]
    fn does_not_split_inside_a_number_or_a_time() {
        assert_eq!(
            clauses("It costs 1,000 dollars at 10:30, give or take a few."),
            ["It costs 1,000 dollars at 10:30,", "give or take a few."]
        );
    }

    #[test]
    fn splits_after_punctuation_inside_closing_quotes() {
        assert_eq!(
            clauses("He said \"wait for me,\" and then he left the room."),
            ["He said \"wait for me,\"", "and then he left the room."]
        );
    }

    #[test]
    fn splits_at_semicolons_colons_and_dashes() {
        assert_eq!(
            clauses(
                "Here is the whole plan: we leave at dawn; we travel light — \
                 and we do not look back."
            ),
            [
                "Here is the whole plan:",
                "we leave at dawn;",
                "we travel light —",
                "and we do not look back.",
            ]
        );
    }

    #[test]
    fn blank_text_yields_no_clauses() {
        assert!(clauses("").is_empty());
        assert!(clauses("  \n ").is_empty());
    }

    /// `lead` samples of silence, `speech` of signal, `tail` of silence.
    fn padded(lead: usize, speech: usize, tail: usize) -> Vec<f32> {
        [vec![0.0; lead], vec![0.5; speech], vec![0.0; tail]].concat()
    }

    #[test]
    fn trims_only_the_joined_edges() {
        let clause = padded(7_200, 1_000, 10_800);
        assert_eq!(trim_joins(&clause, false, false).len(), clause.len());
        assert_eq!(
            trim_joins(&clause, true, false).len(),
            JOIN_LEAD + 1_000 + 10_800
        );
        assert_eq!(
            trim_joins(&clause, false, true).len(),
            7_200 + 1_000 + JOIN_TAIL
        );
        assert_eq!(
            trim_joins(&clause, true, true),
            padded(JOIN_LEAD, 1_000, JOIN_TAIL)
        );
    }

    #[test]
    fn keeps_edges_already_shorter_than_a_join() {
        let clause = padded(100, 1_000, 200);
        assert_eq!(trim_joins(&clause, true, true), clause);
    }

    #[test]
    fn leaves_all_silence_untouched() {
        let silence = vec![0.0; 4_800];
        assert_eq!(trim_joins(&silence, true, true), silence);
    }
}
