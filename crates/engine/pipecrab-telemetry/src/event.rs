//! The raw telemetry [`Event`]: one observer callback, timestamped and owned,
//! cheap to move across the export channel.

use std::sync::Arc;
use std::time::Duration;

use pipecrab_core::{
    DataFrame, Direction, DispatchFrame, Disposition, Finality, ModelFrame, ModelInput,
    ModelMessage, Role, SystemFrame,
};
use pipecrab_runtime::{PerformOutcome, StageId};

/// One observation, in the hub's [`Clock`](crate::Clock) timebase.
#[derive(Clone, Debug)]
pub struct Event {
    /// When the callback fired.
    pub at: Duration,
    /// The stage the callback was about; `None` for pipeline-level events
    /// ([`EventKind::Wired`], [`EventKind::Lost`]).
    pub stage: Option<StageId>,
    /// What was observed.
    pub kind: EventKind,
}

/// The observation itself. `FrameIn`/`Decided` and `SysIn`/`SysDecided` come
/// in adjacent pairs per stage (`decide_*` is synchronous), as do
/// `PerformStart`/`PerformEnd`; durations are the timestamp differences.
#[derive(Clone, Debug)]
pub enum EventKind {
    /// A (sub)pipeline finished wiring; `stages` lists its children in order.
    Wired {
        /// The wired stages, top-level first, nested reported separately with
        /// dotted paths.
        stages: Vec<StageId>,
    },
    /// A data frame reached the stage; its `decide_data` runs next.
    FrameIn(FrameInfo),
    /// The stage's `decide_data` returned.
    Decided {
        /// What happened to the frame.
        disposition: Disposition,
        /// How many effects were queued.
        effects: usize,
    },
    /// A system frame reached the stage.
    SysIn {
        /// The frame's travel direction.
        dir: Direction,
        /// The frame itself (cheaply clonable).
        frame: SystemFrame,
    },
    /// The stage's `decide_system` returned.
    SysDecided {
        /// What happened to the frame.
        disposition: Disposition,
        /// How many effects were queued.
        effects: usize,
    },
    /// The stage is about to perform one effect.
    PerformStart,
    /// The matching perform ended.
    PerformEnd {
        /// How it ended, including `Aborted` on a barge-in.
        outcome: PerformOutcome,
    },
    /// `count` events were dropped because the export channel was full; the
    /// pipeline thread never blocks on telemetry.
    Lost {
        /// How many events were lost since the last successful send.
        count: u64,
    },
}

/// An owned summary of a [`DataFrame`]: everything turn assembly and the
/// fine-tune dataset need, nothing bulky. Text payloads are `Arc` clones;
/// audio is reduced to its dimensions. Partial-transcript text is elided —
/// only its timing and length matter.
#[derive(Clone, Debug)]
pub enum FrameInfo {
    /// Transport input audio, reduced to its dimensions.
    InputAudio {
        /// Payload size in bytes.
        bytes: usize,
        /// Samples per second.
        sample_rate: u32,
        /// Channel count.
        channels: u16,
    },
    /// Decoded audio, reduced to its dimensions (`samples` is the interleaved
    /// total, so seconds = samples / channels / sample_rate).
    Audio {
        /// Interleaved sample count.
        samples: usize,
        /// Samples per second, per channel.
        sample_rate: u32,
        /// Channel count.
        channels: u16,
    },
    /// A transcript. `text` is kept only for [`Finality::Final`].
    Transcript {
        /// The full text, final transcripts only.
        text: Option<Arc<str>>,
        /// Byte length of the text (partials included).
        len: usize,
        /// Who produced it.
        role: Role,
        /// Partial or final.
        finality: Finality,
    },
    /// The user started speaking.
    SpeechStarted,
    /// The user stopped speaking.
    SpeechStopped,
    /// An LM generation began.
    GenerationStarted,
    /// An LM generation completed.
    GenerationFinished,
    /// A complete tool call emitted by the model.
    ToolCall {
        /// Correlates with a later tool result.
        id: Arc<str>,
        /// Model-facing tool name.
        name: Arc<str>,
        /// Complete argument object as JSON text.
        arguments_json: Arc<str>,
    },
    /// A tool result re-entering the model, correlated by `tool_call_id`.
    ToolResult {
        /// The [`FrameInfo::ToolCall`] id this answers.
        tool_call_id: Arc<str>,
        /// The tool's name.
        name: Arc<str>,
        /// The result text.
        content: Arc<str>,
        /// Whether it triggers a generation (`Respond`) or only joins context.
        respond: bool,
    },
    /// A non-user event message entering the model.
    ModelEvent {
        /// The producing system.
        source: Arc<str>,
        /// The event kind, in the producer's vocabulary.
        kind: Arc<str>,
        /// Whether it triggers a generation.
        respond: bool,
    },
    /// A dispatch protocol frame, reduced to its variant name.
    Dispatch {
        /// `"command"` or `"event"`.
        kind: &'static str,
    },
    /// An application-defined frame, reduced to its declared kind.
    Custom {
        /// The frame's [`CustomFrame::kind`](pipecrab_core::CustomFrame::kind).
        kind: &'static str,
    },
}

impl From<&DataFrame> for FrameInfo {
    fn from(frame: &DataFrame) -> Self {
        match frame {
            DataFrame::InputAudio {
                bytes,
                sample_rate,
                num_channels,
            } => FrameInfo::InputAudio {
                bytes: bytes.len(),
                sample_rate: *sample_rate,
                channels: *num_channels,
            },
            DataFrame::Audio(chunk) => FrameInfo::Audio {
                samples: chunk.samples.len(),
                sample_rate: chunk.format.sample_rate,
                channels: chunk.format.channels,
            },
            DataFrame::Transcript(t) => FrameInfo::Transcript {
                text: match t.finality {
                    Finality::Final => Some(t.text.clone()),
                    Finality::Partial { .. } => None,
                },
                len: t.text.len(),
                role: t.role,
                finality: t.finality,
            },
            DataFrame::SpeechStarted => FrameInfo::SpeechStarted,
            DataFrame::SpeechStopped => FrameInfo::SpeechStopped,
            DataFrame::Model(ModelFrame::GenerationStarted) => FrameInfo::GenerationStarted,
            DataFrame::Model(ModelFrame::GenerationFinished) => FrameInfo::GenerationFinished,
            DataFrame::Model(ModelFrame::ToolCall(call)) => FrameInfo::ToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments_json: call.arguments_json.clone(),
            },
            DataFrame::Model(ModelFrame::Input(input)) => {
                let (message, respond) = match input {
                    ModelInput::Context(m) => (m, false),
                    ModelInput::Respond(m) => (m, true),
                };
                match message {
                    ModelMessage::ToolResult {
                        tool_call_id,
                        name,
                        content,
                    } => FrameInfo::ToolResult {
                        tool_call_id: tool_call_id.clone(),
                        name: name.clone(),
                        content: content.clone(),
                        respond,
                    },
                    ModelMessage::Event { source, kind, .. } => FrameInfo::ModelEvent {
                        source: source.clone(),
                        kind: kind.clone(),
                        respond,
                    },
                }
            }
            DataFrame::Dispatch(frame) => FrameInfo::Dispatch {
                kind: match frame {
                    DispatchFrame::Command(_) => "command",
                    DispatchFrame::Event(_) => "event",
                },
            },
            DataFrame::Custom(custom) => FrameInfo::Custom {
                kind: custom.kind(),
            },
        }
    }
}
