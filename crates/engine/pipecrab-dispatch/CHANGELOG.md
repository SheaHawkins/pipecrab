# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-dispatch-v0.5.0...pipecrab-dispatch-v0.5.1) - 2026-07-23

### Added

- dispatch task interface ([#65](https://github.com/SheaHawkins/pipecrab/pull/65))

### Added

- Initial `pipecrab-dispatch` crate: the facade layer connecting native
  `Dispatch` frames to model tool calls and external asynchronous-task
  transports.
- `dispatch_task_definition` / `update_task_definition` / `dispatch_tool_definitions`:
  provider-neutral `pipecrab_lm::ToolDefinition`s for `LmStage::with_tools`.
- `DispatchSource` / `DispatchSink` transport capability traits (with the
  `DispatchTransport` convenience marker); concrete transports are later adapter
  crates.
- `DispatchIngress`: an active pass-through stage whose overridden `Stage::run`
  concurrently polls the system lane, the external source, and the inbound data
  lane, emitting each external event as a raw `Dispatch` frame followed by its
  model projection.
- `DispatchEgress`: translates acknowledged `dispatch_task` / `update_task`
  tool calls into `DispatchCommand`s, sends them through the sink, and emits them
  downstream as native dispatch frames. Tracks only generation-local
  acknowledgement state — no durable task map.
- `Dispatch::new` composition helper: `tool_definitions()` and `into_stages()`.
- `DispatchError` with recoverable/fatal classification and a `StageError`
  conversion.
