use std::any::Any;
use std::sync::Arc;

/// Extension point for application-defined frame payloads.
///
/// Implement this on your own types and wrap them in [`Frame::Custom`] to pass
/// domain-specific data through a pipeline without forking the core frame enum.
pub trait CustomFrame: Any + Send + Sync + std::fmt::Debug {
    /// A static string identifying the concrete frame type (used for logging/dispatch).
    fn kind(&self) -> &'static str;
    /// Downcasting helper; implementations should return `self`.
    fn as_any(&self) -> &dyn Any;
}

/// The unit of data flowing through a pipeline stage.
///
/// Frames travel in either direction (see [`Direction`]). System frames
/// (`Start`, `Stop`, `Interrupt`, `Error`) are handled by every stage;
/// `Audio`, `Transcript`, and `Custom` carry the actual pipeline payload.
/// Immutable: don't try to make mutable frames because it's a sign you're doing something wrong.
/// Instead: Build up a new frame by aggregating other frames and produce it when you're ready.
#[derive(Clone, Debug)]
pub enum Frame {
    /// Pipeline is starting; stages should initialise any runtime state.
    Start,
    /// Graceful shutdown; stages should flush and clean up.
    Stop,
    /// User barged in; stages should discard in-flight work and reset.
    Interrupt,
    /// An error string propagated through the pipeline (usually upstream).
    Error(Arc<str>),
    /// A text transcript segment (ASR output or TTS input).
    Transcript(Arc<str>),
    /// A raw audio chunk (PCM bytes, format negotiated out-of-band).
    Audio(Arc<[u8]>),
    /// Application-defined payload; see [`CustomFrame`].
    Custom(Arc<dyn CustomFrame>),
}

/// Travel direction. Down = source -> sink; Up = sink -> source (errors, acks).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    /// Source → sink (audio/transcript flowing forward through the pipeline).
    Down,
    /// Sink → source (errors, acknowledgements flowing back upstream).
    Up,
}
