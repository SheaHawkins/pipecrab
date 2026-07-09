//! silero-vad: the sans-IO core of pipecrab's Silero voice-activity-detection
//! stack — all of the Silero *model logic*, with no inference backend, no
//! async, and no pipecrab dependency.
//!
//! Silero VAD is a small recurrent speech/non-speech classifier. The subtle,
//! correctness-critical parts live here so the native ([`silero-vad-ort`]) and
//! browser ([`silero-vad-web`]) engines get identical numerics from one
//! implementation: the context prefix, the recurrent-state lifecycle, the
//! streaming/offline segmentation (hysteresis, hangover, padding), and frame
//! validation. An engine crate supplies only the ~20-line inference call; this
//! crate drives it.
//!
//! # Design (locked in `docs/plans/silero-vad.md`, §4.1)
//!
//! This crate is **unimplemented scaffolding**. The interfaces below are the
//! locked design, recorded here so implementation doesn't drift; the code lands
//! in follow-up work.
//!
//! ## Model I/O contract (Silero VAD v6.2.x, v5+ ONNX)
//!
//! | Tensor | Name | Shape / dtype | Notes |
//! |---|---|---|---|
//! | in | `input` | `[1, N+ctx]` f32 | N = 512 @ 16 kHz / 256 @ 8 kHz; ctx = 64 / 32 samples of the previous frame (zeros first) |
//! | in | `state` | `[2, 1, 128]` f32 | zeros on reset |
//! | in | `sr` | scalar i64 | 16000 or 8000 |
//! | out | `output` | `[1, 1]` f32 | speech probability |
//! | out | `stateN` | `[2, 1, 128]` f32 | fed back as `state` |
//!
//! The core is **sync-agnostic by construction**: the native backend is
//! synchronous (`ort::Session::run` takes `&mut self`) while the web backend is
//! inherently async (JS promises), so the core never calls a backend — backends
//! drive it.
//!
//! ## Planned surface
//!
//! - `SampleRate { Hz8000, Hz16000 }` with `frame_len()` (256 | 512) and
//!   `context_len()` (32 | 64); `STATE_LEN = 256` (the `[2, 1, 128]` state,
//!   flattened).
//! - `SileroSession` — owns the recurrent state + context tail, validates
//!   frames (exact `frame_len()`, else `FrameError::InvalidLength`, mirroring the
//!   reference's hard error), and stages the context-prefixed model inputs as a
//!   `Staged<'_>` (`input`, `&mut state`, `sr`). No I/O, no inference.
//! - `SileroModel` — the sync backend trait
//!   (`infer(input, state, sr) -> Result<f32, _>`), plus a `SileroVad<M>`
//!   convenience engine (`process`/`reset`) so standalone sync (native) users
//!   get an ergonomic API without pipecrab or async. Async backends (web) call
//!   `SileroSession::stage` directly across an `.await` and get the same
//!   numerics from the shared staging/state code.
//! - `Segmenter` / `StreamingOptions` (parity with Python `VADIterator`) and
//!   `get_speech_timestamps` / `TimestampOptions` (parity with Python
//!   `get_speech_timestamps`) — pure segmentation, usable with *any* probability
//!   source, so it carries no inference dependency. The Python parameter
//!   vocabulary (`threshold`, `neg_threshold`, `min_silence_duration_ms`,
//!   `speech_pad_ms`, `min_speech_duration_ms`, `max_speech_duration_s`) and its
//!   defaults are kept verbatim — that's the compatibility that matters, since
//!   users' tuning knowledge transfers.
//! - `FrameBuffer` — push arbitrary-length sample slices, pop exact frames — for
//!   *standalone* callers with hardware-sized chunks. Inside pipecrab it is
//!   unused: `VadStage` owns framing (plan §4.5).
//!
//! `reset()` everywhere resets *everything* — state tensor, context, segmenter
//! bookkeeping, frame buffer — per the reference's `reset_states` (a tensor-only
//! reset is a prior-art bug class).
//!
//! ## Cargo
//!
//! No dependencies (std only), so the core is wasm-clean. An optional
//! `bundled-model` feature will embed `silero_vad.onnx` (op16, 2,327,524 bytes,
//! MIT) via `include_bytes!` so both backends share one copy of the bytes —
//! well under crates.io's package limit.
//!
//! ## Correctness bar
//!
//! Frame-level numerical parity with the reference Python `silero-vad` package
//! (v6.2.x), fixture-tested. This is the test prior art fails: skipping the
//! context prefix silently diverges from the reference.
//!
//! [`silero-vad-ort`]: https://docs.rs/silero-vad-ort
//! [`silero-vad-web`]: https://docs.rs/silero-vad-web
#![forbid(unsafe_code)]
#![warn(missing_docs)]
