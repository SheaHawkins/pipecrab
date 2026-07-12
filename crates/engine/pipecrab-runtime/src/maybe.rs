//! `Send` and `Sync` bounds that are vacuous on `wasm32`.

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

/// Applies [`async_trait`](https://docs.rs/async-trait) with `Send` futures on
/// native targets and local futures on `wasm32`.
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
/// The macro reuses this crate's `async-trait` dependency.
#[macro_export]
macro_rules! maybe_async_trait {
    ($item:item) => {
        #[cfg_attr(target_arch = "wasm32", $crate::async_trait::async_trait(?Send))]
        #[cfg_attr(not(target_arch = "wasm32"), $crate::async_trait::async_trait)]
        $item
    };
}
