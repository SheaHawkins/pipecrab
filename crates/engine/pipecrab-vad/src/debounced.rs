//! [`Debounced`]: lifts a raw [`SpeechScorer`] into a full
//! [`VoiceActivityDetector`], the sibling of `pipecrab-stt`'s `Buffered`.
//!
//! A [`SpeechScorer`] answers only "how likely is *this* window speech?" for one
//! exact-length window at a time. [`Debounced`] absorbs, in one place, the three
//! things that stand between that and the edge-emitting
//! [`VoiceActivityDetector`] contract:
//!
//! * **Windowing** — arbitrary-length chunks are accumulated into exact
//!   [`window_len()`](SpeechScorer::window_len) windows, with the remainder
//!   carried across calls.
//! * **Threshold** — a probability is a speech/not-speech bit only once compared
//!   against [`DebounceConfig::threshold`].
//! * **Hangover** — a run of consecutive agreeing windows must accrue before the
//!   state flips ([`start_windows`](DebounceConfig::start_windows) /
//!   [`stop_windows`](DebounceConfig::stop_windows)), so a stray window does not
//!   chatter start/stop pairs.
//!
//! # Only for scorers — double hysteresis is impossible by construction
//!
//! [`Debounced`] is the adapter for *raw scorers only*. A segmenter-class engine
//! (sherpa's VAD) owns its own debounce — its min-speech / min-silence config
//! *is* one — and implements [`VoiceActivityDetector`] directly, never through
//! this adapter. Because a segmenter is never wrapped by `Debounced`, no engine
//! is ever debounced twice: double hysteresis cannot arise by construction.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pipecrab_core::AudioFormat;

use crate::{SpeechScorer, VadError, VadEvent, VoiceActivityDetector};

/// Threshold and hangover for [`Debounced`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebounceConfig {
    /// Probability at or above which a window counts as speech. The default is
    /// `0.5`, silero's conventional midpoint.
    pub threshold: f32,
    /// Consecutive speech windows required to declare speech *started*. Small so
    /// onset is responsive.
    pub start_windows: u32,
    /// Consecutive non-speech windows required to declare speech *stopped*.
    /// Larger, so a brief pause mid-utterance does not clip it.
    pub stop_windows: u32,
}

impl Default for DebounceConfig {
    fn default() -> Self {
        // React to onset immediately; ride out short gaps before closing.
        Self {
            threshold: 0.5,
            start_windows: 1,
            stop_windows: 8,
        }
    }
}

/// The current run of the edge detector: are we in speech, and how many
/// consecutive windows have disagreed with that so far. Ported verbatim from the
/// stage's former `VadState`.
struct ObserveState {
    in_speech: bool,
    against: u32,
}

impl ObserveState {
    /// Feed one window's `is_speech` verdict; return a [`VadEvent`] if it flips
    /// the state once the [`DebounceConfig`] hangover is satisfied.
    fn observe(&mut self, is_speech: bool, config: &DebounceConfig) -> Option<VadEvent> {
        if is_speech == self.in_speech {
            // Verdict agrees with the current state: reset the opposing run.
            self.against = 0;
            return None;
        }
        self.against += 1;
        let needed = if self.in_speech {
            config.stop_windows
        } else {
            config.start_windows
        };
        if self.against < needed {
            return None;
        }
        self.in_speech = is_speech;
        self.against = 0;
        Some(if is_speech {
            VadEvent::SpeechStarted
        } else {
            VadEvent::SpeechStopped
        })
    }

    fn reset(&mut self) {
        self.in_speech = false;
        self.against = 0;
    }
}

/// The mutable session state, behind a [`Mutex`] because the trait methods take
/// `&self`. The lock is never held across an `.await`.
struct DebouncedState {
    /// Samples that did not fill a whole window; carried to the next call.
    remainder: Vec<f32>,
    /// The hangover state machine.
    observe: ObserveState,
}

/// Lifts a [`SpeechScorer`] into a [`VoiceActivityDetector`] by windowing its
/// input, thresholding its probabilities, and debouncing the result into edges.
/// The sibling of `pipecrab-stt`'s `Buffered`; see the module docs.
///
/// Arbitrary input chunks may split or contain several scorer windows, so this
/// adapter intentionally copies samples into exact, contiguous windows. Direct
/// [`VoiceActivityDetector`] implementations receive the pipeline's shared
/// sample buffer without that windowing copy.
pub struct Debounced<S: SpeechScorer> {
    scorer: S,
    config: DebounceConfig,
    state: Mutex<DebouncedState>,
}

impl<S: SpeechScorer> Debounced<S> {
    /// Wrap `scorer` with the default [`DebounceConfig`].
    pub fn new(scorer: S) -> Self {
        Self::with_config(scorer, DebounceConfig::default())
    }

    /// Wrap `scorer` with an explicit [`DebounceConfig`].
    pub fn with_config(scorer: S, config: DebounceConfig) -> Self {
        Self {
            scorer,
            config,
            state: Mutex::new(DebouncedState {
                remainder: Vec::new(),
                observe: ObserveState {
                    in_speech: false,
                    against: 0,
                },
            }),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<S: SpeechScorer> VoiceActivityDetector for Debounced<S> {
    fn input_format(&self) -> AudioFormat {
        self.scorer.input_format()
    }

    async fn process(&self, samples: Arc<[f32]>) -> Result<Vec<VadEvent>, VadError> {
        let window_len = self.scorer.window_len();
        // Lock → append the new samples, extract every complete window into a
        // local Vec, unlock. The remainder (a partial window) stays safe in
        // state. A `process` dropped after this point loses only the locally
        // extracted windows — the edge is delayed by a window or two, never
        // resurrected — and the remainder is untouched.
        let windows: Vec<Arc<[f32]>> = {
            let mut st = self.state.lock().expect("Debounced state mutex poisoned");
            st.remainder.extend_from_slice(&samples);
            let mut windows = Vec::new();
            while st.remainder.len() >= window_len && window_len > 0 {
                windows.push(Arc::from(
                    st.remainder.drain(..window_len).collect::<Vec<_>>(),
                ));
            }
            windows
        };

        // Score each window with no lock held, then take a short lock to observe
        // the verdict and collect any edge.
        let mut events = Vec::new();
        for window in windows {
            let prob = self.scorer.score(window).await?;
            let is_speech = prob >= self.config.threshold;
            let mut st = self.state.lock().expect("Debounced state mutex poisoned");
            if let Some(event) = st.observe.observe(is_speech, &self.config) {
                events.push(event);
            }
        }
        Ok(events)
    }

    fn reset(&self) {
        let mut st = self.state.lock().expect("Debounced state mutex poisoned");
        st.remainder.clear();
        st.observe.reset();
    }
}
