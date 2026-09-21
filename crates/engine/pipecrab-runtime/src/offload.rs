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

/// # wasm
///
/// `wasm32-unknown-unknown` has no [`std::thread`], so there is nowhere to send
/// `f`: it runs inline on the orchestrator, bracketed by a yield either side so
/// a queued system frame is still seen before and after the call. The bounds are
/// unchanged, so one call site compiles on both targets.
///
/// This bracket is not a substitute for a thread — `f` still occupies the
/// orchestrator for its whole duration. Work heavy enough to need one belongs in
/// a JS Worker behind its capability trait, which is where the browser engines
/// put it (`Transcriber` awaits the Worker; the stage stays engine-neutral).
#[cfg(target_arch = "wasm32")]
pub async fn offload<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    yield_now().await;
    let result = f();
    yield_now().await;
    result
}

/// Return `Pending` exactly once, after scheduling an immediate re-poll.
#[cfg(target_arch = "wasm32")]
async fn yield_now() {
    let mut yielded = false;
    core::future::poll_fn(move |cx| {
        if yielded {
            return core::task::Poll::Ready(());
        }
        yielded = true;
        cx.waker().wake_by_ref();
        core::task::Poll::Pending
    })
    .await
}
