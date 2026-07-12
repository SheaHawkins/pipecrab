//! Adapts a raw [`SpeechScorer`] into a [`VoiceActivityDetector`].
//!
//! [`Debounced`] supplies the policy a scorer lacks:
//!
//! * It buffers arbitrary input into [`SpeechScorer::window_len`] windows.
//! * It compares probabilities with [`DebounceConfig::threshold`].
//! * It requires consecutive windows before changing state.
//!
//! Segmenting engines that already debounce should implement
//! [`VoiceActivityDetector`] directly.

use std::sync::Mutex;

use async_trait::async_trait;
use pipecrab_core::AudioFormat;

use crate::{SpeechScorer, VadError, VadEvent, VoiceActivityDetector};

/// Threshold and hangover for [`Debounced`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebounceConfig {
    /// Probability at or above which a window counts as speech.
    pub threshold: f32,
    /// Consecutive speech windows required to start speech.
    pub start_windows: u32,
    /// Consecutive non-speech windows required to stop speech.
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

/// Current speech state and consecutive disagreeing windows.
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

/// Mutable session state. The lock is never held across an await point.
struct DebouncedState {
    /// Samples that did not fill a whole window; carried to the next call.
    remainder: Vec<f32>,
    /// The hangover state machine.
    observe: ObserveState,
}

/// Windows scorer input and debounces its probabilities into speech edges.
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

    async fn process(&self, samples: &[f32]) -> Result<Vec<VadEvent>, VadError> {
        let window_len = self.scorer.window_len();
        // Lock → append the new samples, extract every complete window into a
        // local Vec, unlock. The remainder (a partial window) stays safe in
        // state. A `process` dropped after this point loses only the locally
        // extracted windows — the edge is delayed by a window or two, never
        // resurrected — and the remainder is untouched.
        let windows: Vec<Vec<f32>> = {
            let mut st = self.state.lock().expect("Debounced state mutex poisoned");
            st.remainder.extend_from_slice(samples);
            let mut windows = Vec::new();
            while st.remainder.len() >= window_len && window_len > 0 {
                windows.push(st.remainder.drain(..window_len).collect());
            }
            windows
        };

        // Score each window with no lock held, then take a short lock to observe
        // the verdict and collect any edge.
        let mut events = Vec::new();
        for window in windows {
            let prob = self.scorer.score(&window).await?;
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
