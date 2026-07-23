//! In-memory scripted source and recording sink shared by the integration
//! tests. Deterministic and hardware-free, driven by `block_on`.
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::stream::StreamExt;
use pipecrab_core::{DispatchCommand, DispatchEvent};
use pipecrab_dispatch::{DispatchError, DispatchSink, DispatchSource};

/// One programmed outcome of a [`ScriptedSource::next_event`] call. Dropping the
/// paired [`mpsc::Sender`] closes the source (yields `Ok(None)`).
pub enum SourceStep {
    /// The source yields this event.
    Event(DispatchEvent),
    /// The source returns this error.
    Fail(DispatchError),
}

/// A [`DispatchSource`] whose events (and errors) the test feeds over a channel,
/// with a shared cancel counter.
///
/// Deliberately `Send` but not `Sync` (an mpsc `Receiver` is `!Sync`): it
/// exercises `DispatchIngress`'s ownership model, where a source need not be
/// `Sync`.
pub struct ScriptedSource {
    rx: mpsc::Receiver<SourceStep>,
    cancels: Arc<AtomicUsize>,
}

impl ScriptedSource {
    /// A source plus the sender that drives it and the shared cancel counter.
    pub fn new() -> (Self, mpsc::Sender<SourceStep>, Arc<AtomicUsize>) {
        let (tx, rx) = mpsc::channel(16);
        let cancels = Arc::new(AtomicUsize::new(0));
        (
            Self {
                rx,
                cancels: cancels.clone(),
            },
            tx,
            cancels,
        )
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl DispatchSource for ScriptedSource {
    async fn next_event(&mut self) -> Result<Option<DispatchEvent>, DispatchError> {
        match self.rx.next().await {
            Some(SourceStep::Event(event)) => Ok(Some(event)),
            Some(SourceStep::Fail(error)) => Err(error),
            None => Ok(None),
        }
    }

    fn cancel(&self) {
        self.cancels.fetch_add(1, Ordering::SeqCst);
    }
}

/// A [`DispatchSink`] that records the commands it is sent, or fails every send.
#[derive(Clone)]
pub struct RecordingSink {
    sent: Arc<Mutex<Vec<DispatchCommand>>>,
    fail: Option<DispatchError>,
}

impl RecordingSink {
    /// A sink that records commands, plus a handle to the recorded log.
    pub fn new() -> (Self, Arc<Mutex<Vec<DispatchCommand>>>) {
        let sent = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                sent: sent.clone(),
                fail: None,
            },
            sent,
        )
    }

    /// A sink that returns `error` on every send (recording nothing).
    pub fn failing(error: DispatchError) -> (Self, Arc<Mutex<Vec<DispatchCommand>>>) {
        let sent = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                sent: sent.clone(),
                fail: Some(error),
            },
            sent,
        )
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl DispatchSink for RecordingSink {
    async fn send_command(&self, command: DispatchCommand) -> Result<(), DispatchError> {
        if let Some(error) = &self.fail {
            return Err(error.clone());
        }
        self.sent.lock().unwrap().push(command);
        Ok(())
    }
}
