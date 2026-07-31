# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release: a ZeroClaw gateway webhook transport for `pipecrab-dispatch`.
  `connect(ZeroclawConfig)` returns a `ZeroclawSource` / `ZeroclawSink` pair;
  each command spawns a detached worker for one synchronous `POST /webhook`
  turn — no poller, since the gateway answers only when the turn finishes.
  - `dispatch_task` mints a `pc-{uuid}` task id and sends it as `X-Session-Id`,
    with the tool call id as `X-Idempotency-Key`. `Accepted` is emitted as the
    request departs (there is no cheap acceptance signal to await), so
    gateway-level refusals surface as `Failure`s.
  - The worker maps the reply: a 2xx `response` → `Completion`, a 4xx → a
    non-retryable `Failure`, a 429 / 5xx / transport error → a retryable one,
    a `needs_quickstart` 503 → non-retryable, and a `duplicate` dedupe body →
    retryable (the original reply is unrecoverable). No `Progress` or
    `Question` events — one request is one opaque turn.
  - `update_task` posts a follow-up under the same session id; ZeroClaw's
    session memory carries the conversation server-side, so no history is
    replayed. An unknown task, or one whose turn is still executing, is
    `Rejected`.
  - `cancel()` closes the source and abandons in-flight requests, deliberately
    leaving the gateway to finish its turns — external task state outlives the
    turn.
