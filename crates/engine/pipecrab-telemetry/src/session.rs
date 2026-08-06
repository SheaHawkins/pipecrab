//! [`Telemetry`] and [`Session`]: the native convenience layer that owns the
//! worker thread. (On `wasm32` drive [`TurnAssembler`](crate::TurnAssembler)
//! from the hub's receiver yourself; there is no thread to spawn.)
//!
//! ```no_run
//! use std::sync::Arc;
//! use pipecrab_telemetry::{Telemetry, TelemetrySink, SinkError, TurnRecord};
//!
//! struct Print;
//! impl TelemetrySink for Print {
//!     fn record(&mut self, turn: &TurnRecord) -> Result<(), SinkError> {
//!         println!("turn {}: {:?} -> {:?}", turn.turn, turn.user_text, turn.agent_text);
//!         Ok(())
//!     }
//! }
//!
//! let telemetry = Telemetry::builder().sink(Print).build();
//! let session = telemetry.session();
//! let observer = session.observer(); // give to PipelineBuilder::observer(..)
//! // ... run the pipeline to completion ...
//! session.finish();
//! ```

use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use pipecrab_runtime::StageObserver;

use crate::assembler::TurnAssembler;
use crate::clock::{Clock, StdClock};
use crate::event::Event;
use crate::hub::{DEFAULT_CAPACITY, TelemetryHub};
use crate::record::SessionInfo;
use crate::sink::TelemetrySink;

/// Builds a [`Telemetry`]: the sinks and (optionally) the clock and channel
/// capacity every session created from it will use.
pub struct TelemetryBuilder {
    sinks: Vec<Box<dyn TelemetrySink + Send>>,
    capacity: usize,
}

impl TelemetryBuilder {
    /// Add a sink; every finished turn record is offered to each sink in the
    /// order added.
    pub fn sink(mut self, sink: impl TelemetrySink + Send + 'static) -> Self {
        self.sinks.push(Box::new(sink));
        self
    }

    /// Override the hub's event-channel bound (default
    /// [`DEFAULT_CAPACITY`]).
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(1);
        self
    }

    /// Finish building.
    pub fn build(self) -> Telemetry {
        Telemetry {
            sinks: self.sinks,
            capacity: self.capacity,
        }
    }
}

/// The telemetry configuration an app builds once; [`session`](Telemetry::session)
/// mints the per-interaction [`Session`].
pub struct Telemetry {
    sinks: Vec<Box<dyn TelemetrySink + Send>>,
    capacity: usize,
}

impl Telemetry {
    /// Start building; add sinks with [`TelemetryBuilder::sink`].
    pub fn builder() -> TelemetryBuilder {
        TelemetryBuilder {
            sinks: Vec::new(),
            capacity: DEFAULT_CAPACITY,
        }
    }

    /// Open a session with a generated id (hex Unix nanos). Consumes the
    /// telemetry configuration — one session per interaction, one pipeline
    /// per session.
    pub fn session(self) -> Session {
        let id = format!(
            "{:x}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        self.session_with_id(id)
    }

    /// Open a session under an application-chosen id.
    pub fn session_with_id(self, id: impl Into<Arc<str>>) -> Session {
        let clock = Arc::new(StdClock::new());
        let started_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let session = SessionInfo {
            id: id.into(),
            started_unix_ms,
        };
        let (hub, rx) = TelemetryHub::with_capacity(clock.clone(), self.capacity);
        let worker = spawn_worker(rx, TurnAssembler::new(session.clone()), self.sinks, clock);
        Session {
            info: session,
            hub: Some(hub),
            worker: Some(worker),
        }
    }
}

/// One observed interaction: holds the hub the pipeline reports into and the
/// worker thread that assembles and sinks its records.
pub struct Session {
    info: SessionInfo,
    hub: Option<Arc<TelemetryHub>>,
    worker: Option<JoinHandle<()>>,
}

impl Session {
    /// This session's identity.
    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    /// The observer to hand to
    /// [`PipelineBuilder::observer`](pipecrab_runtime::PipelineBuilder::observer).
    ///
    /// # Panics
    ///
    /// Panics if called after [`finish`](Session::finish).
    pub fn observer(&self) -> Arc<dyn StageObserver> {
        self.hub
            .as_ref()
            .expect("session already finished")
            .observer()
    }

    /// End the session: flush the open turn and every sink, then join the
    /// worker. Call this **after** the pipeline driver has completed — the
    /// worker exits when the last observer handle (the pipeline's included)
    /// is dropped, so finishing earlier blocks until the pipeline is done.
    pub fn finish(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        // Drop our hub reference; once the pipeline's clones are gone too, the
        // channel disconnects and the worker drains, flushes, and exits.
        self.hub = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if self.worker.is_some() {
            self.shutdown();
        }
    }
}

fn spawn_worker(
    rx: Receiver<Event>,
    mut assembler: TurnAssembler,
    mut sinks: Vec<Box<dyn TelemetrySink + Send>>,
    clock: Arc<StdClock>,
) -> JoinHandle<()> {
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Duration;

    /// How often the assembler ticks while the event stream is quiet, so a
    /// finished turn's record lands promptly (see `TurnAssembler::tick`).
    const TICK: Duration = Duration::from_millis(200);

    std::thread::Builder::new()
        .name("pipecrab-telemetry".into())
        .spawn(move || {
            let offer = |turn: &crate::record::TurnRecord,
                         sinks: &mut Vec<Box<dyn TelemetrySink + Send>>| {
                for sink in sinks.iter_mut() {
                    if let Err(error) = sink.record(turn) {
                        eprintln!("pipecrab-telemetry: {error}");
                    }
                }
            };
            loop {
                let closed = match rx.recv_timeout(TICK) {
                    Ok(event) => assembler.push(event),
                    Err(RecvTimeoutError::Timeout) => assembler.tick(clock.now()),
                    Err(RecvTimeoutError::Disconnected) => break,
                };
                if let Some(turn) = closed {
                    offer(&turn, &mut sinks);
                }
            }
            if let Some(turn) = assembler.finish() {
                offer(&turn, &mut sinks);
            }
            for sink in sinks.iter_mut() {
                if let Err(error) = sink.flush() {
                    eprintln!("pipecrab-telemetry: {error}");
                }
            }
        })
        .expect("spawn telemetry worker")
}
