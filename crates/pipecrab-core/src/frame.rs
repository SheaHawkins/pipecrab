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
/// These are bidirectional: `Interrupt` and `Start`/`Stop` travel downstream;
/// `Error` typically travels upstream. Immutable once constructed.
#[derive(Clone, Debug)]
pub enum SystemFrame {
    /// Pipeline is starting; stages should initialise any runtime state.
    Start,
    /// Graceful shutdown; stages should flush and clean up.
    Stop,
    /// User barged in; stages should discard in-flight work and reset.
    Interrupt,
    /// An error propagated through the pipeline.
    Error {
        /// Human-readable description of the error.
        message: Arc<str>,
        /// Whether the error is unrecoverable and the pipeline should shut down.
        fatal: bool,
    },
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
    /// A text transcript segment (ASR output or TTS input).
    Transcript(Arc<str>),
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
    /// use pipecrab_core::{AudioChunk, AudioFormat, DataFrame};
    ///
    /// let input = DataFrame::InputAudio {
    ///     bytes: Arc::from(&[0u8; 4][..]),
    ///     sample_rate: 16_000,
    ///     num_channels: 1,
    /// };
    /// assert!(input.survives_flush());
    ///
    /// assert!(!DataFrame::Transcript("hi".into()).survives_flush());
    ///
    /// let audio = AudioChunk::new(Arc::from(&[0.0f32][..]), AudioFormat::new(48_000, 1));
    /// assert!(!DataFrame::Audio(audio).survives_flush());
    /// ```
    pub fn survives_flush(&self) -> bool {
        matches!(self, DataFrame::InputAudio { .. })
    }
}
