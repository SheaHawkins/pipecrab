//! `offload` runs work off the orchestrator thread and returns its result.
//! Native-only: the wasm path is an unimplemented stub.
#![cfg(not(target_arch = "wasm32"))]

use futures::executor::block_on;
use pipecrab_runtime::offload;

#[test]
fn offload_runs_off_thread_and_returns_result() {
    let main = std::thread::current().id();

    let (worker, sum) = block_on(offload(move || {
        // A little CPU-bound work, plus which thread ran it.
        (std::thread::current().id(), (1..=1_000u64).sum::<u64>())
    }));

    assert_eq!(sum, 500_500, "offload returns the closure's result");
    assert_ne!(worker, main, "the closure runs off the orchestrator thread");
}
