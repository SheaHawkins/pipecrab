# Server mode: shared engine pools

## Goal

Server mode runs many concurrent voice sessions on one host. Today each
pipeline owns its engines outright: one `LlamaCpp` worker per session pins a
full model context, one `SherpaStt` actor per session pins a recognizer. That
is the right shape on a phone and the wrong shape on a server, where N
sessions must share M engine workers per capability, M < N.

Add `pipecrab-pool`: an engine-agnostic sharing layer. A pool owns a fleet of
existing engine handles; each session gets a *lease* that implements the same
capability trait (`LanguageModel`, `StreamingTranscriber`, `Synthesizer`).
Per-session pipelines do not change at all — a lease drops into `LmStage`,
`SttStage`, or `TtsStage` exactly where a dedicated engine went.

Multiple shared LLMs are multiple pools: one pool per model, each handing out
its own leases. Which session gets a lease from which pool is app wiring, so
no registry type is needed.

## Why the traits already permit this

Session state lives in the stages, not the engines. `LmStage` owns the
`Conversation` and hands the full context to every `generate` call;
`SttStage` scopes engine state to one VAD-bracketed utterance; `synthesize`
is stateless per call. So a shared engine serving session A then session B is
already *correct*:

- `LlamaCpp`'s KV prefix reuse degrades gracefully — a conversation switch
  shrinks the shared prefix and re-prefills, costing latency, not
  correctness. Slot affinity (below) recovers the fast path.
- A `SherpaStt` utterance is self-contained: begin, feed, end, stream
  dropped.

Three things block sharing today, and they are what this crate exists to fix:

1. **Cancel is handle-global.** Every engine's `cancel()` bumps one epoch for
   the whole worker. Sharing a raw handle means one session's barge-in kills
   another session's in-flight generation.
2. **`StreamingTranscriber` allows one active utterance per engine**, by
   protocol. Two sessions feeding one raw handle is a protocol violation.
3. **No admission control.** Raw handles queue commands unboundedly on the
   worker channel, with no fairness and no backpressure signal.

## Scope

This change includes:

- A publishable `crates/engine/pipecrab-pool` crate, generic over the
  capability traits — no engine crate dependency.
- `LmPool<M>`, `SttPool<T>`, `TtsPool<S>` and their leases `LmLease`,
  `SttLease`, `TtsLease`, each lease implementing the corresponding trait.
- FIFO admission with a bounded waiter queue and fail-fast overflow.
- Per-lease cancellation that can never reach another session's work.
- LM slot affinity so an unbroken session keeps hitting its KV prefix cache.
- Deterministic mock-engine tests; no model files, no network.

This change does not include:

- Network transport, session lifecycle, or a server binary (follow-up plan;
  the cpal bridge already anticipates "a server spawns one pump per
  session").
- Continuous batching (llama.cpp multi-sequence decode in one context). The
  pool is a fleet of single-sequence contexts; a batched scheduler is a
  later, drop-in `LanguageModel` implementation.
- Cross-slot conversation migration via `save_state`/`load_state`.
- Sherpa multi-stream batch decoding for STT.
- Pooling hosted LM adapters — a hosted provider is already concurrent;
  wrap it in a pool of size N only if you want its admission control.
- Metrics export, autoscaling, GPU placement.

## Repository placement

```toml
[package.metadata.pipecrab]
layer = "facade"
```

Like `pipecrab-dispatch`, the crate sits above the trait crates and below
adapters: it depends on `pipecrab-lm`, `pipecrab-stt`, `pipecrab-tts`,
`pipecrab-runtime` (for `MaybeSend` bounds), and `futures` — nothing else.
No threads, no timers, no executor, so it wasm-checks like every other
engine crate; CI gains one line:

```console
cargo check -p pipecrab-pool --target wasm32-unknown-unknown
```

## Public API

```rust
pub struct PoolConfig {
    /// Waiters allowed behind a fully busy fleet before acquires fail fast.
    pub max_waiters: usize, // default 0: never queue, fail immediately
}

pub struct LmPool<M: LanguageModel>;
pub struct SttPool<T: StreamingTranscriber>;
pub struct TtsPool<S: Synthesizer>;

impl<M: LanguageModel> LmPool<M> {
    pub fn new(engines: Vec<M>, config: PoolConfig) -> Result<Self, PoolBuildError>;
    pub fn lease(&self) -> LmLease<M>;
}
// SttPool / TtsPool mirror this shape.

pub struct LmLease<M>;  // implements LanguageModel
pub struct SttLease<T>; // implements StreamingTranscriber
pub struct TtsLease<S>; // implements Synthesizer
```

Pools are cheap-clone handles over shared state. A lease is per-session and
not `Clone`: it is the unit of cancellation isolation, one per stage.

Construction validates the fleet: `new` rejects an empty `Vec`, `SttPool`
rejects engines whose `input_format()` disagree, `TtsPool` likewise for
`output_format()`. The lease answers the sync format methods from the value
cached at construction, so they stay callable from `decide_*`.

`LmLease::save_state`/`load_state` return `LmError::Engine` describing the
limitation: a slot's KV state belongs to whichever conversation ran last, so
checkpointing through a pool is meaningless until the migration follow-up.

## Admission

One `Mutex<PoolState>` guards slot ownership and the waiter queue; every
critical section is O(1) bookkeeping, so control calls stay non-blocking.

A slot is `Free` or `Held(lease_id)`. Acquire, in order: the lease's sticky
slot if free (LM only), any free slot, else enqueue a `oneshot` waiter if
the queue holds fewer than `max_waiters`, else fail now with an error naming
the pool and its depth. Release hands the slot to the oldest live waiter or
marks it free.

Fail-fast is deliberate: a voice session has a dead-air budget, and waiting
behind a deep queue is a worse experience than an immediate error the app
can turn into "all agents busy". `max_waiters` is the only knob; there is no
timed wait because the runtime has no timer primitive and the pool must not
bake in an executor.

Hold spans — how long one lease occupies a slot:

| Capability | Acquired at | Released at |
| --- | --- | --- |
| LM | `generate` | model stream terminates or is dropped |
| STT | `begin_utterance` | `end_utterance` returns, or `cancel` |
| TTS | `synthesize` | audio stream terminates or is dropped |

The returned LM/TTS streams are wrapped: items pass through untouched; a
drop guard releases the slot. The STT span matches the VAD gate — sessions
hold a recognizer only while speech is actually arriving, which is the duty
cycle that makes M < N work.

## Cancellation isolation

A lease's `cancel()` does two things under the pool lock:

1. Tombstones its queued waiter, if any — an abandoned acquire never touches
   an engine.
2. Forwards the inner engine's `cancel()` **only if this lease currently
   holds the slot**.

The inner cancel is still engine-global, but the holder check makes the
blast radius exactly the caller's own work. The check and the slot handoff
happen under the same lock, so a cancel can never race a release and land on
the next session's generation.

Early teardown always cancels before releasing: when a wrapped stream is
dropped mid-flight, the guard forwards inner `cancel()` first, then releases
the slot. Without that ordering, a llama.cpp worker keeps decoding until its
next send fails, and the following holder's first command would queue behind
the stale tail.

Stale output cannot cross sessions for the same reason it cannot cross
utterances today: every engine already epoch-guards its own replies, and a
new holder's call snapshots a fresh epoch after the previous holder's cancel.

## LM slot affinity

Each slot remembers the last lease it served. Acquire prefers that slot when
free, so an unbroken conversation keeps landing on the context whose KV
cache holds its prefix, and per-turn prefill stays proportional to the new
tokens. A miss is correctness-neutral: `LlamaCpp` sees a shorter shared
prefix and re-prefills. Affinity is a selection policy only — no state moves
between slots.

## Sizing guidance (non-normative)

Slots per pool track concurrent *activity*, not sessions: LM slots ≈ replies
being generated at once, STT slots ≈ utterances in flight, TTS slots ≈
replies being spoken. Conversational duty cycles put each well under one per
session; benchmarks on server hardware, not this plan, pick the ratios.

## Tests

Mock engines with channel-controlled blocking points make every race
deterministic:

- Two sessions interleave over a one-slot pool; each sees only its own
  deltas.
- FIFO: waiters are served in arrival order.
- Overflow past `max_waiters` fails immediately; the engine is never
  touched.
- Cancel isolation: A cancels while B generates on the other slot — B's
  stream is unaffected and A's engine records the only inner cancel.
- Cancel while queued abandons the waiter; a later release skips the
  tombstone and wakes the next live waiter.
- Dropping an LM/TTS stream mid-flight forwards inner cancel, then frees the
  slot for the next waiter.
- STT: the slot is held from `begin_utterance` through `end_utterance`;
  protocol errors from the engine pass through unchanged; `cancel` releases.
- Sticky acquire returns a lease to its previous slot when free, and falls
  back to any free slot otherwise.
- Mixed-format fleets are rejected at construction.
- `save_state`/`load_state` on a lease fail with the documented error.
- Calls on a lease after its pool is dropped fail as `Engine` errors rather
  than hanging.

## Verification

```console
cargo fmt --all -- --check
cargo test -p pipecrab-pool
cargo test -p pipecrab-arch --test layering
cargo clippy -p pipecrab-pool --all-targets -- -D warnings
cargo check -p pipecrab-pool --target wasm32-unknown-unknown
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Acceptance criteria

- A lease drops into `LmStage`/`SttStage`/`TtsStage` wherever a dedicated
  engine went; no engine or stage crate changes.
- N pipelines run against M < N engines per capability with correct output
  interleaving.
- One session's `cancel` can never abort or corrupt another session's
  in-flight work, queued or running.
- Admission is FIFO, bounded by `max_waiters`, and fails fast past the bound.
- An unbroken conversation re-acquires its previous LM slot when it is free.
- Two LM pools over different models coexist, each serving its own leases.
- All pool behavior is tested deterministically with mock engines.

## Server-mode roadmap (follow-up plans)

1. **Transport & sessions** — a server frontend (WebSocket first) that
   spawns one pipeline task and one audio pump pair per connection, wired to
   pool leases; lands with an `examples/server-voice-agent`.
2. **Continuous batching** — a llama.cpp multi-sequence scheduler
   implementing `LanguageModel` once per *model*, replacing the
   fleet-of-contexts for LM at higher session density.
3. **Conversation migration** — `save_state`/`load_state` slot swapping so a
   hot conversation can follow its lease across slots and hosts.
4. **STT batch decode** — Sherpa multi-stream decoding to collapse the STT
   fleet into one recognizer.
5. **Linux TTS verification** — the README's remaining ❌ before a Linux
   server target is honest.
