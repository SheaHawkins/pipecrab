# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release: a Hermes Agent runs-API transport for `pipecrab-dispatch`.
  `connect(HermesConfig)` returns a `HermesSource` / `HermesSink` pair and spawns
  a background poller.
  - `dispatch_task` posts `POST /v1/runs` with an adapter-minted `session_id`
    (the `task_id` the model holds) and the tool call id as `Idempotency-Key`.
  - The poller maps run status onto dispatch events: `completed` → `Completion`,
    `failed` → non-retryable `Failure`, `cancelled` → retryable `Failure`, a 404
    on a tracked run → expired-state `Failure`, and any other status change → a
    deduped `Progress`. Network blips and 5xx/401 responses are retried silently.
  - `update_task` chains a new run into the same session under the same
    `task_id`; an unknown task, or one whose run is still executing, is
    `Rejected`.
  - `cancel()` stops the poller and closes the source, deliberately leaving
    remote runs running — external task state outlives the turn.
