//! [`offload`](fn@self::offload) runs CPU-bound or blocking work off the
//! orchestrator thread.
//!
//! The pipeline runs on a single `!Send` thread, and a stage's `perform` must
//! keep yielding so an interrupt can preempt it. Heavy or blocking work run
//! inline would freeze the whole pipeline; `offload` is the one place work
//! crosses to another thread. That is why `F` and `T` are `Send + 'static` —
//! the bound is the offload boundary, not a requirement on the pipeline itself.

/// Run `f` off the orchestrator thread and `await` its result.
///
/// Wrap CPU-bound or blocking work in `offload(...)` and `.await` it: the
/// orchestrator stays free to keep polling — including the system lane — while
/// the work runs elsewhere, so interrupt barge-in stays responsive.
///
/// # Native
///
/// Runs `f` on a fresh [`std::thread`] and returns its result over a
/// `futures::channel::oneshot`. Runtime-agnostic and tokio-free; a runtime
/// adapter can later back this with a pooled `spawn_blocking`.
///
/// Dropping the returned future before it resolves detaches the worker thread —
/// it still runs to completion, but its result is discarded. If `f` panics, the
/// worker unwinds and drops the sender; awaiting the returned future then panics
/// (a fresh panic noting the worker produced no result — the original payload is
/// not propagated).
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

/// wasm stub: offloading to a Web Worker is not yet implemented.
///
/// `wasm32-unknown-unknown` has no `std::thread`, so this placeholder keeps the
/// crate compiling for wasm. The eventual implementation will post `f` to a Web
/// Worker and resolve over a `oneshot`; see the native version for the intended
/// semantics.
#[cfg(target_arch = "wasm32")]
pub async fn offload<F, T>(_f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    unimplemented!("offload on wasm32 (Web Worker path) is not yet implemented")
}
