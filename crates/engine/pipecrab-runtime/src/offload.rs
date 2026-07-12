//! Runs blocking or CPU-bound work outside the pipeline task.
//!
//! Stage effects must yield so system frames can preempt them. [`offload()`]
//! moves work that would block those yields across the thread boundary, which
//! is why its closure and result require `Send + 'static`.

/// Run `f` off the orchestrator thread and `await` its result.
///
/// On native targets, runs `f` on a new [`std::thread`].
///
/// Dropping the future detaches the worker. A worker panic causes the awaiting
/// task to panic without preserving the original payload.
#[cfg(not(target_arch = "wasm32"))]
pub async fn offload<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.await
        .expect("offload worker panicked or was dropped before sending a result")
}

/// Panics because Web Worker offloading is not implemented.
#[cfg(target_arch = "wasm32")]
pub async fn offload<F, T>(_f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    unimplemented!("offload on wasm32 (Web Worker path) is not yet implemented")
}
