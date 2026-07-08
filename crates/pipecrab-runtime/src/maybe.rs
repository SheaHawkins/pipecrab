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

/// Apply [`async_trait`](https://docs.rs/async-trait) with the target-correct
/// `Send`-ness, in one line instead of the two-attribute `cfg_attr` dance.
///
/// Every async trait definition and impl in pipecrab needs the same pair —
/// plain `#[async_trait]` on native (its boxed futures are `Send`, matching the
/// [`MaybeSend`]/[`MaybeSendSync`] bounds), and `#[async_trait(?Send)]` on
/// `wasm32` (where they can't be). Writing both by hand is easy to get subtly
/// wrong (swap the two `cfg`s and native silently loses `Send`). Wrap the item
/// instead — the trait definition *or* the impl block:
///
/// ```
/// use pipecrab_runtime::{maybe_async_trait, MaybeSend};
///
/// maybe_async_trait! {
///     pub trait Widget: MaybeSend {
///         async fn poll(&mut self) -> u32;
///     }
/// }
///
/// struct Zero;
/// maybe_async_trait! {
///     impl Widget for Zero {
///         async fn poll(&mut self) -> u32 { 0 }
///     }
/// }
/// ```
///
/// The macro pulls in `async_trait` through this crate, so a stage or interface crate
/// that uses it needs only a `pipecrab-runtime` dependency, not a direct one on
/// `async-trait`.
#[macro_export]
macro_rules! maybe_async_trait {
    ($item:item) => {
        #[cfg_attr(target_arch = "wasm32", $crate::async_trait::async_trait(?Send))]
        #[cfg_attr(not(target_arch = "wasm32"), $crate::async_trait::async_trait)]
        $item
    };
}
