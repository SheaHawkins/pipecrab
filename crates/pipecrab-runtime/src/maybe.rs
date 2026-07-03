//! Target-conditional `Send`/`Sync` bounds: real on native, vacuous on wasm32.
//!
//! The pipeline runs as *one logical task* on every target. On native that task
//! should still be `Send` so a work-stealing executor (e.g. multi-threaded
//! tokio) can spawn and migrate it like any other task. On `wasm32` there is
//! only one thread and JS handles (`JsValue`) are `!Send`, so the same bounds
//! must vanish. These aliases express "Send where it exists" once, so the rest
//! of the codebase is written a single time.

/// `Send` on native targets; no requirement on `wasm32`.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: ?Sized + Send> MaybeSend for T {}

/// `Send` on native targets; no requirement on `wasm32`.
#[cfg(target_arch = "wasm32")]
pub trait MaybeSend {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSend for T {}

/// `Send + Sync` on native targets; no requirement on `wasm32`.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSendSync: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: ?Sized + Send + Sync> MaybeSendSync for T {}

/// `Send + Sync` on native targets; no requirement on `wasm32`.
#[cfg(target_arch = "wasm32")]
pub trait MaybeSendSync {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSendSync for T {}
