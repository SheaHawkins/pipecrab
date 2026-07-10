use std::any::Any;
use std::sync::Arc;

/// Extension point for application-defined frame payloads.
///
/// Implement this on your own types and wrap them in [`DataFrame::Custom`] to
/// pass domain-specific data through a pipeline without forking the core frame
/// enum.
pub trait CustomFrame: Any + Send + Sync + std::fmt::Debug {
    /// A static string identifying the concrete frame type (used for logging/dispatch).
    fn kind(&self) -> &'static str;
    /// Downcasting helper; implementations should return `self`.
    fn as_any(&self) -> &dyn Any;
}

/// The wire format of an [`AudioChunk`]: its sample rate and channel count.
///
/// Samples are always `f32`; only the rate and channel count vary along the
/// pipeline (capture ~48 kHz → STT 16 kHz → TTS 24 kHz → playback ~48 kHz),
/// which is why every chunk carries its own format instead of assuming one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AudioFormat {
    /// Samples per second, per channel (e.g. 48 000 for 48 kHz).
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo). Samples are interleaved.
    pub channels: u16,
}

impl AudioFormat {
    /// Construct a format from a `sample_rate` and `channels` count.
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self { sample_rate, channels }
    }
}

/// A chunk of `f32` PCM audio tagged with its own [`AudioFormat`].
///
/// Immutable like every [`DataFrame`]: aggregate chunks and produce a new one
/// rather than mutating in place. `samples` are interleaved by channel; for the
/// common mono case that is just a flat sample buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioChunk {
    /// Interleaved `f32` PCM samples.
    pub samples: Arc<[f32]>,
    /// The rate and channel count these `samples` are in.
    pub format: AudioFormat,
}

impl AudioChunk {
    /// Bundle `samples` with the `format` they are in.
    pub fn new(samples: Arc<[f32]>, format: AudioFormat) -> Self {
        Self { samples, format }
    }
}

/// Travel direction for system frames.
///
/// Down = source → sink; Up = sink → source (errors, acks).
/// [`DataFrame`] carries no direction — media is always downstream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    /// Source → sink (lifecycle, interrupts flowing forward through the pipeline).
    Down,
    /// Sink → source (errors, acknowledgements flowing back upstream).
    Up,
}

/// System frames: lifecycle, control, and errors.
///
/// These are bidirectional: `Interrupt`, `Start`/`Stop`, and the
/// `SpeechStarted`/`SpeechStopped` voice-activity edges travel downstream;
/// `Error` typically travels upstream. Immutable once constructed.
#[derive(Clone, Debug)]
pub enum SystemFrame {
    /// Pipeline is starting; stages should initialise any runtime state.
    Start,
    /// Graceful shutdown; stages should flush and clean up.
    Stop,
    /// User barged in; stages should discard in-flight work and reset.
    Interrupt,
    /// Voice-activity detection observed the user *start* speaking. Travels
    /// downstream so stages can open an utterance and prepare to transcribe.
    /// Emitted by a VAD stage on the silence→speech edge, not per audio window.
    SpeechStarted,
    /// Voice-activity detection observed the user *stop* speaking. Travels
    /// downstream so stages can close the utterance and flush it for
    /// transcription. Emitted on the speech→silence edge.
    SpeechStopped,
    /// An error propagated through the pipeline.
    Error {
        /// Human-readable description of the error.
        message: Arc<str>,
        /// Whether the error is unrecoverable and the pipeline should shut down.
        fatal: bool,
    },
}

/// A piece of conversation text flowing through the pipeline: speech-to-text
/// output, language-model output, or text bound for TTS.
///
/// # The stable-prefix invariant
///
/// For a given utterance (a [`Role::User`] unit) or generation (a
/// [`Role::Agent`] unit), the bytes `text[..stable]` **never change** across
/// successive partials: once a prefix is declared stable it is frozen, and only
/// the tail beyond `stable` may be revised or grow. Downstream stages rely on
/// this to commit settled text early instead of waiting for
/// [`Finality::Final`].
///
/// `stable` is a byte index into `text` and must lie on a char boundary. STT
/// partials revise their tail, so `stable <= text.len()`; LM partials are
/// append-only, so `stable == text.len()`. A [`Finality::Final`] transcript
/// carries no `stable` — the whole `text` is settled.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    /// The transcript text so far. For a [`Finality::Partial`] this is the
    /// current best guess; only `text[..stable]` is guaranteed frozen.
    pub text: Arc<str>,
    /// Who produced this text — [`Role::User`] (STT) or [`Role::Agent`] (LM).
    pub role: Role,
    /// Whether this is an in-progress [`Finality::Partial`] or the completed
    /// [`Finality::Final`] unit.
    pub finality: Finality,
}

/// Who produced a [`Transcript`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Speech-to-text output: what the user said.
    User,
    /// Language-model output: what the agent is saying.
    Agent,
}

/// Whether a [`Transcript`] is still being revised or is complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finality {
    /// In-progress text. `stable` is the byte length of the prefix that will
    /// never change. STT partials revise their tail (`stable <= text.len()`);
    /// LM partials are append-only (`stable == text.len()`).
    Partial {
        /// Byte length of the frozen prefix. Lies on a char boundary and is
        /// `<= text.len()`; see the [`Transcript`] stable-prefix invariant.
        stable: usize,
    },
    /// The completed unit (utterance for [`Role::User`], generation for
    /// [`Role::Agent`] — but see the `SentenceChunker` note in Part 4).
    Final,
}

impl Transcript {
    /// An in-progress user (STT) transcript: the first `stable` bytes of `text`
    /// are frozen and its tail may still be revised.
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
        Self { text, role: Role::User, finality: Finality::Partial { stable } }
    }

    /// A completed user utterance.
    pub fn user_final(text: impl Into<Arc<str>>) -> Self {
        Self { text: text.into(), role: Role::User, finality: Finality::Final }
    }

    /// An in-progress agent (LM) transcript. LM output is append-only, so the
    /// entire current `text` is stable (`stable == text.len()`).
    pub fn agent_partial(text: impl Into<Arc<str>>) -> Self {
        let text = text.into();
        let stable = text.len();
        Self { text, role: Role::Agent, finality: Finality::Partial { stable } }
    }

    /// A completed agent generation.
    pub fn agent_final(text: impl Into<Arc<str>>) -> Self {
        Self { text: text.into(), role: Role::Agent, finality: Finality::Final }
    }
}

/// Data frames: media payload flowing downstream (source → sink).
///
/// Immutable: don't try to make mutable frames. Instead, aggregate frames and
/// produce a new one when you're ready.
#[derive(Clone, Debug)]
pub enum DataFrame {
    /// Input audio from a transport source. Survives an interrupt flush so that
    /// a barge-in utterance is not clipped; see [`DataFrame::survives_flush`].
    InputAudio {
        /// Raw PCM bytes.
        bytes: Arc<[u8]>,
        /// Samples per second (e.g. 16 000 for 16 kHz).
        sample_rate: u32,
        /// Number of audio channels (1 = mono, 2 = stereo).
        num_channels: u16,
    },
    /// A piece of conversation text: STT output, LM output, or text bound for
    /// TTS. See [`Transcript`] for the role and finality it carries.
    Transcript(Transcript),
    /// A chunk of `f32` PCM audio carrying its own [`AudioFormat`].
    Audio(AudioChunk),
    /// Application-defined payload; see [`CustomFrame`].
    Custom(Arc<dyn CustomFrame>),
}

impl DataFrame {
    /// True for frames that must survive an interrupt's data-queue flush —
    /// input-from-transport media, since a barge-in utterance must not be
    /// clipped. False for everything else.
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
    /// assert!(!DataFrame::Transcript(Transcript::agent_final("hi")).survives_flush());
    ///
    /// let audio = AudioChunk::new(Arc::from(&[0.0f32][..]), AudioFormat::new(48_000, 1));
    /// assert!(!DataFrame::Audio(audio).survives_flush());
    /// ```
    pub fn survives_flush(&self) -> bool {
        matches!(self, DataFrame::InputAudio { .. })
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
        assert_eq!(ap.finality, Finality::Partial { stable: "hi there".len() });

        let af = Transcript::agent_final("done");
        assert_eq!(af.role, Role::Agent);
        assert_eq!(af.finality, Finality::Final);
    }
}
