//! silero-vad-ort: the native onnxruntime (`ort`) engine for Silero VAD.
//!
//! This is the host backend behind
//! [`pipecrab-vad-silero`](https://docs.rs/pipecrab-vad-silero)'s
//! `VoiceActivityDetector` impl; the browser counterpart is
//! [`silero-vad-web`](https://docs.rs/silero-vad-web). It supplies the inference
//! call for the shared [`silero-vad`](https://docs.rs/silero-vad) core
//! (implementing that crate's `SileroModel` trait); all of the Silero model
//! logic — context prefix, state lifecycle, segmentation — lives in the core,
//! not here.
//!
//! # Design (locked in `docs/plans/silero-vad.md`, §4.2 and §5)
//!
//! This crate is **unimplemented scaffolding**; the decisions below are recorded
//! so implementation doesn't drift.
//!
//! ## Planned surface
//!
//! - `OrtModel` wrapping an `ort::Session`, with `from_session` (BYO — the app
//!   fully controls ort setup), `from_bytes`, `from_file`, and `bundled()`
//!   (behind `bundled-model`) constructors, implementing
//!   `silero_vad::SileroModel` via zero-copy `TensorRef` inputs.
//! - `type SileroVadOrt = silero_vad::SileroVad<OrtModel>`, plus `pub use ort;`
//!   and `pub use silero_vad;` so apps configure linking / execution providers
//!   against *our* ort version and standalone users get one import.
//! - Session defaults: CPU only (GPU transfer overhead exceeds compute for a
//!   ~1 ms/frame model), `with_intra_threads(1)`.
//!
//! ## ort / ONNX Runtime linking — the app decides, never this crate
//!
//! ```toml
//! [dependencies]
//! ort = { version = "2.0.0-rc.12", default-features = false, features = ["std"] }
//!
//! [features]
//! default = ["bundled-model"]
//! bundled-model = ["silero-vad/bundled-model"]
//! # passthroughs so apps can opt in through us if convenient:
//! download-binaries = ["ort/download-binaries", "ort/tls-native"]
//! load-dynamic = ["ort/load-dynamic"]
//! ```
//!
//! - **Caret, never `=`**: `ort-sys` declares `links = "onnxruntime"`, so two
//!   exact pins on different rcs are a *hard build error* across the whole
//!   dependency graph. A caret bound unifies rc.12 with later rcs and 2.0.0
//!   final — the wound prior art keeps open, which we close.
//! - **`default-features = false`**: ort's default `download-binaries` (build-
//!   time static fetch) is the right *application* default, but features are
//!   additive, so a *library* enabling it (or `load-dynamic`) would lock every
//!   downstream app into that linking mode. The app picks exactly one:
//!   `download-binaries` (static, self-contained), `load-dynamic` (runtime
//!   dlopen — air-gapped / system-lib path), or an alternative backend; the
//!   passthrough features above let it opt in through us.
//! - **BYO `Session` (`from_session`)**: the escape hatch that makes this crate
//!   agnostic even to *how* the session was built.
//! - **Binary size** (documented, not chosen by us): the ONNX Runtime CPU build
//!   dwarfs the 2.2 MB model — ~15–30 MB statically linked; apps that care use
//!   `load-dynamic` or a reduced-ops source build.
//! - **MSRV**: ort rc.12 requires Rust 1.88, above the workspace floor (1.75),
//!   so this crate (and native builds of `pipecrab-vad-silero`) will carry a
//!   per-crate `rust-version = "1.88"` override rather than raise the floor.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
