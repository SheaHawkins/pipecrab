//! The [`Clock`] a hub timestamps with: monotonic, relative to session start.

use std::time::Duration;

use pipecrab_runtime::MaybeSendSync;

/// A monotonic clock. `now()` is the elapsed time since the clock (and so the
/// session) began; every [`Event`](crate::Event) timestamp and every
/// [`TurnRecord`](crate::TurnRecord) offset is in this timebase.
pub trait Clock: MaybeSendSync {
    /// Monotonic elapsed time since the clock's epoch.
    fn now(&self) -> Duration;
}

/// [`Clock`] over [`std::time::Instant`], epoch = construction time.
/// Native-only: `Instant::now()` traps on `wasm32-unknown-unknown`.
#[cfg(not(target_arch = "wasm32"))]
pub struct StdClock {
    epoch: std::time::Instant,
}

#[cfg(not(target_arch = "wasm32"))]
impl StdClock {
    /// A clock whose epoch is now.
    pub fn new() -> Self {
        Self {
            epoch: std::time::Instant::now(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for StdClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Clock for StdClock {
    fn now(&self) -> Duration {
        self.epoch.elapsed()
    }
}
