use std::any::Any;
use std::sync::Arc;

/// An application-defined [`DataFrame::Custom`] payload.
pub trait CustomFrame: Any + Send + Sync + std::fmt::Debug {
    /// Identifies the concrete frame type for logging or dispatch.
    fn kind(&self) -> &'static str;
    /// Returns `self` for downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// The sample rate and channel count of an [`AudioChunk`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AudioFormat {
    /// Samples per second per channel.
    pub sample_rate: u32,
    /// Number of interleaved channels.
    pub channels: u16,
}

impl AudioFormat {
    /// Creates a format.
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
        }
    }
}

/// A chunk of interleaved `f32` PCM audio.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioChunk {
    /// Interleaved `f32` PCM samples.
    pub samples: Arc<[f32]>,
    /// The samples' format.
    pub format: AudioFormat,
}

impl AudioChunk {
    /// Creates an audio chunk.
    pub fn new(samples: Arc<[f32]>, format: AudioFormat) -> Self {
        Self { samples, format }
    }
}

/// The travel direction of a [`SystemFrame`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    /// Source to sink.
    Down,
    /// Sink to source.
    Up,
}

/// Priority lifecycle, control, and error frames.
///
/// These frames may overtake queued data. Use [`DataFrame`] for events that
/// mark a position in the media stream and must preserve FIFO order.
#[derive(Clone, Debug)]
pub enum SystemFrame {
    /// Starts the pipeline.
    Start,
    /// Stops the pipeline gracefully.
    Stop,
    /// Cancels in-flight work after a user barge-in.
    Interrupt,
    /// An error propagated through the pipeline.
    Error {
        /// Human-readable description of the error.
        message: Arc<str>,
        /// Whether the error is unrecoverable and the pipeline should shut down.
        fatal: bool,
    },
}

/// Conversation text produced by speech recognition or a language model.
///
/// # The stable-prefix invariant
///
/// Across successive partials, `text[..stable]` never changes. Only the
/// remaining suffix may change or grow.
///
/// `stable` is a character-boundary byte index. It is at most `text.len()` for
/// user text and equals `text.len()` for append-only agent text.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    /// The current text. For [`Finality::Partial`], only `text[..stable]` is fixed.
    pub text: Arc<str>,
    /// Who produced the text.
    pub role: Role,
    /// Whether the text is partial or final.
    pub finality: Finality,
}

/// Who produced a [`Transcript`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Speech-to-text output.
    User,
    /// Language-model output.
    Agent,
}

/// Whether a [`Transcript`] is still being revised or is complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finality {
    /// In-progress text with a stable prefix.
    Partial {
        /// Byte length of the fixed prefix. See [`Transcript`].
        stable: usize,
    },
    /// A completed utterance or generation.
    Final,
}

impl Transcript {
    /// Creates a partial user transcript.
    ///
    /// `stable` must be `<= text.len()` and lie on a char boundary; both are
    /// debug-asserted.
    pub fn user_partial(text: impl Into<Arc<str>>, stable: usize) -> Self {
        let text = text.into();
        debug_assert!(
            stable <= text.len(),
            "stable byte index {stable} exceeds text length {}",
            text.len()
        );
        debug_assert!(
            text.is_char_boundary(stable),
            "stable byte index {stable} is not on a char boundary of {text:?}"
        );
        Self {
            text,
            role: Role::User,
            finality: Finality::Partial { stable },
        }
    }

    /// A completed user utterance.
    pub fn user_final(text: impl Into<Arc<str>>) -> Self {
        Self {
            text: text.into(),
            role: Role::User,
            finality: Finality::Final,
        }
    }

    /// Creates an append-only partial agent transcript.
    pub fn agent_partial(text: impl Into<Arc<str>>) -> Self {
        let text = text.into();
        let stable = text.len();
        Self {
            text,
            role: Role::Agent,
            finality: Finality::Partial { stable },
        }
    }

    /// A completed agent generation.
    pub fn agent_final(text: impl Into<Arc<str>>) -> Self {
        Self {
            text: text.into(),
            role: Role::Agent,
            finality: Finality::Final,
        }
    }
}

/// Frames that flow downstream in FIFO order.
#[derive(Clone, Debug)]
pub enum DataFrame {
    /// Transport audio that survives interrupt flushes.
    InputAudio {
        /// Raw PCM bytes.
        bytes: Arc<[u8]>,
        /// Samples per second (e.g. 16 000 for 16 kHz).
        sample_rate: u32,
        /// Number of audio channels (1 = mono, 2 = stereo).
        num_channels: u16,
    },
    /// Conversation text. See [`Transcript`].
    Transcript(Transcript),
    /// A chunk of `f32` PCM audio carrying its own [`AudioFormat`].
    Audio(AudioChunk),
    /// A speech-start edge preceding the utterance's [`Audio`](Self::Audio).
    SpeechStarted,
    /// A speech-stop edge following the utterance's last [`Audio`](Self::Audio).
    SpeechStopped,
    /// Application-defined payload; see [`CustomFrame`].
    Custom(Arc<dyn CustomFrame>),
}

impl DataFrame {
    /// Returns whether this frame survives an interrupt's data-queue flush.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use pipecrab_core::{AudioChunk, AudioFormat, DataFrame, Transcript};
    ///
    /// let input = DataFrame::InputAudio {
    ///     bytes: Arc::from(&[0u8; 4][..]),
    ///     sample_rate: 16_000,
    ///     num_channels: 1,
    /// };
    /// assert!(input.survives_flush());
    ///
    /// assert!(!DataFrame::from(Transcript::agent_final("hi")).survives_flush());
    ///
    /// let audio = AudioChunk::new(Arc::from(&[0.0f32][..]), AudioFormat::new(48_000, 1));
    /// assert!(!DataFrame::Audio(audio).survives_flush());
    /// ```
    pub fn survives_flush(&self) -> bool {
        matches!(self, DataFrame::InputAudio { .. })
    }
}

impl From<Transcript> for DataFrame {
    /// Wraps a [`Transcript`] in [`DataFrame::Transcript`].
    fn from(transcript: Transcript) -> Self {
        DataFrame::Transcript(transcript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_does_not_survive_flush() {
        // A transcript is derived, regenerable output — unlike transport input
        // audio it must not survive an interrupt's data-lane flush, in any of
        // its role/finality forms.
        for t in [
            Transcript::user_partial("partial", 3),
            Transcript::user_final("done"),
            Transcript::agent_partial("streaming"),
            Transcript::agent_final("said"),
        ] {
            assert!(!DataFrame::Transcript(t).survives_flush());
        }
    }

    #[test]
    fn voice_edges_do_not_survive_flush() {
        // The VAD edges ride the data lane but are derived control, not captured
        // media: a barge-in flush discards them, same as a transcript.
        assert!(!DataFrame::SpeechStarted.survives_flush());
        assert!(!DataFrame::SpeechStopped.survives_flush());
    }

    #[test]
    fn constructors_set_role_finality_and_stable() {
        let up = Transcript::user_partial("hello", 3);
        assert_eq!(up.role, Role::User);
        assert_eq!(up.finality, Finality::Partial { stable: 3 });

        let uf = Transcript::user_final("hello");
        assert_eq!(uf.role, Role::User);
        assert_eq!(uf.finality, Finality::Final);

        // LM partials are append-only: the whole text is stable.
        let ap = Transcript::agent_partial("hi there");
        assert_eq!(ap.role, Role::Agent);
        assert_eq!(
            ap.finality,
            Finality::Partial {
                stable: "hi there".len()
            }
        );

        let af = Transcript::agent_final("done");
        assert_eq!(af.role, Role::Agent);
        assert_eq!(af.finality, Finality::Final);
    }

    #[test]
    fn user_partial_accepts_stable_on_char_boundaries() {
        // "héllo": 'é' occupies bytes 1..3, so byte 3 (start of the first 'l')
        // is a valid interior boundary; 0 and text.len() are the trivial ones.
        for stable in [0usize, 3, "héllo".len()] {
            let t = Transcript::user_partial("héllo", stable);
            assert_eq!(t.finality, Finality::Partial { stable });
        }
    }

    // The `stable` invariant is enforced by `debug_assert!`, so the failure
    // cases only panic in debug builds; gate them so `cargo test --release`
    // (asserts compiled out) does not expect a panic that never fires.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "exceeds text length")]
    fn user_partial_rejects_stable_past_end() {
        // stable = 3 > "hi".len() = 2.
        let _ = Transcript::user_partial("hi", 3);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "char boundary")]
    fn user_partial_rejects_stable_off_char_boundary() {
        // "é" is two UTF-8 bytes; stable = 1 splits the codepoint.
        let _ = Transcript::user_partial("é", 1);
    }

    #[test]
    fn transcript_converts_into_dataframe() {
        let frame: DataFrame = Transcript::user_final("hi").into();
        match frame {
            DataFrame::Transcript(t) => {
                assert_eq!(&*t.text, "hi");
                assert_eq!(t.role, Role::User);
                assert_eq!(t.finality, Finality::Final);
            }
            other => panic!("expected a Transcript frame, got {other:?}"),
        }
    }
}
