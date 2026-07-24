//! [`HermesSource`]: the receive half. A thin wrapper over the event channel's
//! receiver, so [`next_event`](DispatchSource::next_event) is cancellation-safe
//! by construction — dropping an unresolved poll of a tokio `Receiver` loses no
//! buffered event.

use std::sync::Arc;

use async_trait::async_trait;
use pipecrab_core::DispatchEvent;
use pipecrab_dispatch::{DispatchError, DispatchSource};
use tokio::sync::mpsc;

use crate::inner::Inner;

/// The receive half of the Hermes transport: the poller and sink emit onto a
/// bounded channel, and this drains it. Owned single-threaded by
/// [`DispatchIngress`](pipecrab_dispatch::DispatchIngress) — `Send`, not `Sync`,
/// exactly as the trait allows.
pub struct HermesSource {
    inner: Arc<Inner>,
    events: mpsc::Receiver<DispatchEvent>,
}

impl HermesSource {
    pub(crate) fn new(inner: Arc<Inner>, events: mpsc::Receiver<DispatchEvent>) -> Self {
        Self { inner, events }
    }
}

#[async_trait]
impl DispatchSource for HermesSource {
    async fn next_event(&mut self) -> Result<Option<DispatchEvent>, DispatchError> {
        // `recv` is cancellation-safe; `None` (all senders dropped, i.e. the
        // poller stopped after `cancel`) is a graceful close.
        Ok(self.events.recv().await)
    }

    fn cancel(&self) {
        self.inner.cancel();
    }
}
