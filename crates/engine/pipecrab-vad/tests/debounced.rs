//! `Debounced` lifts a raw `SpeechScorer` (per-window probabilities) into a
//! `VoiceActivityDetector` (speech edges). It owns the windowing (arbitrary
//! chunks → exact windows, remainder carried across calls), the threshold, and
//! the hangover. These tests pin all three plus `reset` and the alternation
//! invariant, using a `MockScorer` of scripted probabilities.
//!
//! Deterministic and tokio-free (`block_on`), so it rides the default
//! `cargo test --workspace` path.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::executor::block_on;
use pipecrab_core::AudioFormat;
use pipecrab_vad::{
    DebounceConfig, Debounced, SpeechScorer, VadError, VadEvent, VoiceActivityDetector,
};

/// A hardware-free scorer: returns a scripted probability per `score` call and
/// records the length of every window it was handed, so windowing is verifiable.
struct MockScorer {
    probs: Mutex<VecDeque<f32>>,
    window_len: usize,
    scored_lengths: Arc<Mutex<Vec<usize>>>,
}

impl MockScorer {
    fn new(window_len: usize, probs: Vec<f32>) -> Self {
        Self {
            probs: Mutex::new(probs.into_iter().collect()),
            window_len,
            scored_lengths: Arc::default(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl SpeechScorer for MockScorer {
    fn input_format(&self) -> AudioFormat {
        AudioFormat::new(16_000, 1)
    }

    fn window_len(&self) -> usize {
        self.window_len
    }

    async fn score(&self, window: &[f32]) -> Result<f32, VadError> {
        self.scored_lengths.lock().unwrap().push(window.len());
        Ok(self.probs.lock().unwrap().pop_front().unwrap_or(0.0))
    }
}

fn config(threshold: f32, start: u32, stop: u32) -> DebounceConfig {
    DebounceConfig {
        threshold,
        start_windows: start,
        stop_windows: stop,
    }
}

#[test]
fn windows_odd_chunks_and_carries_the_remainder_across_calls() {
    block_on(async {
        // Window length 4; enough scripted 0.0s that no edge fires (we only care
        // about windowing here).
        let scorer = MockScorer::new(4, vec![0.0; 8]);
        let scored = scorer.scored_lengths.clone();
        let vad = Debounced::with_config(scorer, config(0.5, 1, 1));

        // 3 samples: remainder 3, no complete window yet.
        assert!(vad.process(&[0.0; 3]).await.unwrap().is_empty());
        assert_eq!(
            *scored.lock().unwrap(),
            Vec::<usize>::new(),
            "no window from 3 < 4 samples"
        );

        // +3 -> remainder 6 -> one window of 4, remainder 2.
        vad.process(&[0.0; 3]).await.unwrap();
        // +5 -> remainder 7 -> one window of 4, remainder 3.
        vad.process(&[0.0; 5]).await.unwrap();
        // +1 -> remainder 4 -> one window of 4, remainder 0.
        vad.process(&[0.0; 1]).await.unwrap();

        // Three windows extracted across the calls, each exactly window_len.
        assert_eq!(
            *scored.lock().unwrap(),
            vec![4, 4, 4],
            "every window is exactly window_len"
        );
    });
}

#[test]
fn threshold_is_inclusive_at_the_boundary() {
    block_on(async {
        // start_windows = 1, so a single speech window fires SpeechStarted. A
        // probability exactly at the threshold counts as speech (>=).
        let at = Debounced::with_config(MockScorer::new(2, vec![0.5]), config(0.5, 1, 1));
        assert_eq!(
            at.process(&[0.0; 2]).await.unwrap(),
            vec![VadEvent::SpeechStarted],
            "0.5 >= 0.5"
        );

        // Just below the threshold is silence: no edge.
        let below = Debounced::with_config(MockScorer::new(2, vec![0.499]), config(0.5, 1, 1));
        assert!(
            below.process(&[0.0; 2]).await.unwrap().is_empty(),
            "0.499 < 0.5 is not speech"
        );
    });
}

#[test]
fn edges_debounce_with_hangover_and_alternate() {
    block_on(async {
        // start after 2 speech windows, stop after 3 silence windows. The lone
        // dip mid-speech must be ridden out by the stop hangover.
        let probs = vec![
            0.1, // silence
            0.9, 0.9, // 2 speech -> SpeechStarted
            0.1, 0.1, // 2 silence < stop hangover of 3: no edge
            0.9, // back to speech: resets the stop run
            0.1, 0.1, 0.1, // 3 silence -> SpeechStopped
        ];
        let vad = Debounced::with_config(MockScorer::new(2, probs.clone()), config(0.5, 2, 3));

        // Feed one window (2 samples) per scripted probability, collecting edges.
        let mut events = Vec::new();
        for _ in 0..probs.len() {
            events.extend(vad.process(&[0.0; 2]).await.unwrap());
        }
        assert_eq!(
            events,
            vec![VadEvent::SpeechStarted, VadEvent::SpeechStopped],
            "one clean start/stop pair; events alternate starting with Started",
        );
    });
}

#[test]
fn reset_clears_both_the_accumulator_and_the_observe_state() {
    block_on(async {
        // window_len 4, start after 1 speech window.
        let scorer = MockScorer::new(4, vec![0.9, 0.9]);
        let scored = scorer.scored_lengths.clone();
        let vad = Debounced::with_config(scorer, config(0.5, 1, 1));

        // Leave a 2-sample remainder and move the state into speech.
        assert_eq!(
            vad.process(&[0.0; 6]).await.unwrap(),
            vec![VadEvent::SpeechStarted]
        );
        // The window scored was exactly 4; a 2-sample remainder is carried.
        assert_eq!(*scored.lock().unwrap(), vec![4]);

        vad.reset();

        // Accumulator cleared: the carried 2 samples are gone, so 2 fresh samples
        // do NOT complete a window (they would have, 2 + 2 = 4, without the reset).
        assert!(
            vad.process(&[0.0; 2]).await.unwrap().is_empty(),
            "the remainder was cleared"
        );
        assert_eq!(
            *scored.lock().unwrap(),
            vec![4],
            "no new window scored after reset"
        );

        // Observe state cleared: we are idle again, so the next speech window
        // re-fires SpeechStarted rather than staying silent.
        assert_eq!(
            vad.process(&[0.0; 2]).await.unwrap(),
            vec![VadEvent::SpeechStarted],
            "reset returned the state to idle, so onset fires afresh",
        );
    });
}
