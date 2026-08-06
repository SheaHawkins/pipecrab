//! pipecrab-telemetry: sessions, traces (turns), and spans for a pipecrab
//! pipeline, assembled from the frames the pipeline already carries.
//!
//! The frame vocabulary *is* the semantic event stream — `SpeechStarted` /
//! `SpeechStopped`, transcript finality, `GenerationStarted` / `Finished`,
//! complete `ToolCall`s and their correlated tool results — so no trace
//! context is threaded through frames. Instead:
//!
//! - a [`TelemetryHub`] implements the runtime's
//!   [`StageObserver`](pipecrab_runtime::StageObserver): each callback becomes
//!   a timestamped [`Event`] pushed through a bounded, non-blocking channel
//!   (the pipeline thread never blocks on telemetry; overflow drops events and
//!   reports the loss);
//! - a [`TurnAssembler`] folds those events into [`TurnRecord`]s — one per
//!   back-and-forth turn, carrying the user text, agent text, tool calls, and
//!   the latency metrics that are only clean to measure in-process
//!   (`time_to_first_speech` above all);
//! - [`TelemetrySink`]s are pluggable encoders of the finished records (JSONL
//!   for fine-tune datasets, OTLP for dashboards); one assembler feeds every
//!   sink, so there is a single source of truth.
//!
//! # Hierarchy
//!
//! - **Session** — one pipeline run / interaction; the app mints it via
//!   [`Telemetry::session`]. All records carry its id.
//! - **Trace (turn)** — opens on the user's `SpeechStarted` (or on
//!   `GenerationStarted` for model-initiated speech) and closes when the next
//!   turn opens, an `Interrupt` cuts it short, or the session ends.
//! - **Spans** — per-stage decide/perform timings aggregated per turn.
//!
//! # Timekeeping
//!
//! The runtime is clock-free; all timestamps come from the hub's [`Clock`].
//! [`StdClock`] (native) is monotonic via `Instant`. On `wasm32` supply your
//! own `Clock` (e.g. over `performance.now()`) — `Instant::now()` traps there.
//! The native [`Session`] worker thread is likewise `cfg`-ed off on wasm;
//! drive [`TurnAssembler`] yourself from the hub's receiver instead.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod assembler;
pub mod clock;
pub mod event;
pub mod hub;
pub mod record;
#[cfg(not(target_arch = "wasm32"))]
pub mod session;
pub mod sink;

pub use assembler::TurnAssembler;
pub use clock::Clock;
#[cfg(not(target_arch = "wasm32"))]
pub use clock::StdClock;
pub use event::{Event, EventKind, FrameInfo};
pub use hub::TelemetryHub;
pub use record::{
    SessionInfo, StageTimings, ToolCallRecord, TurnEnd, TurnOrigin, TurnRecord, TurnTimings,
};
#[cfg(not(target_arch = "wasm32"))]
pub use session::{Session, Telemetry, TelemetryBuilder};
pub use sink::{SinkError, TelemetrySink};
