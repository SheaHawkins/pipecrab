//! The detached poller task: on an interval (or a prompt wake), it sweeps every
//! active run once and lets [`Inner`] fold the results into events.
//!
//! It must be a *detached* task, not inline work in the source: ingress drops
//! and recreates the source's `next_event` future on every passing frame, so an
//! inline `sleep-then-GET` would restart its sleep each time and, under
//! inter-frame gaps shorter than the interval, never poll at all. A standalone
//! task polls on its own clock regardless of ingress churn.

use std::sync::Arc;

use tokio::time::{MissedTickBehavior, interval};

use crate::inner::Inner;

/// Run until cancelled or the event channel closes. Ownership of this `Arc` is
/// the poller's; dropping it on exit removes one reference to [`Inner`].
pub(crate) async fn run(inner: Arc<Inner>) {
    let mut ticker = interval(inner.poll_interval());
    // A backpressured tick must not stampede once the pipeline drains.
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        // Wake on either the interval or an explicit nudge (cancel / new run).
        tokio::select! {
            biased;
            () = inner.woken() => {}
            _ = ticker.tick() => {}
        }

        if inner.is_cancelled() {
            break;
        }
        // A closed channel means the source is gone: stop polling.
        if inner.poll_once().await.is_err() {
            break;
        }
    }
}
