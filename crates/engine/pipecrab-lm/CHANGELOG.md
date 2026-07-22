# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Structured, tool-aware generation. `ModelDelta` (text or a complete `ToolCall`)
  and `ModelStream` replace `TokenOut`/`TokenStream`; `LmStage` translates the
  stream into native `ModelFrame`s — agent transcripts for text,
  `ModelFrame::ToolCall` for calls, bracketed by `GenerationStarted`/`Finished`.
- Provider-neutral `ToolDefinition` (JSON Schema `parameters`).
  `LmStage::with_tools`/`add_tools` configure tools on the stage, duplicate-checked
  and passed to every generation. An adapter wrapping a higher-level agent keeps
  its own registered tools internal.
- `ModelInput::Context` (append without generating) and `ModelInput::Respond`
  (append and generate) input paths on `LmStage`.
- `LmConfigError`, and tool/stream variants on `LmError`.

### Changed

- `LanguageModel::generate` takes the effective `&[ToolDefinition]`.
- `Message` is a structured enum (system/user/assistant-with-tool-calls/tool-result/event),
  replacing the `ChatRole` struct, so conversation history preserves tool calls,
  tool results, and events. Removed `ChatRole`, `TokenOut`, `TokenStream`.

## [0.4.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-lm-v0.3.0...pipecrab-lm-v0.4.0) - 2026-07-17

### Other

- update msrv, rust edition ([#44](https://github.com/SheaHawkins/pipecrab/pull/44))

## [0.3.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-lm-v0.2.0...pipecrab-lm-v0.3.0) - 2026-07-13

### Other

- format repo
- group crates into engine/adapters/support with a layering gate ([#37](https://github.com/SheaHawkins/pipecrab/pull/37))

## [0.2.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-lm-v0.1.0...pipecrab-lm-v0.2.0) - 2026-07-12

### Other

- LmStage, LanguageModel for interruptable LM  ([#33](https://github.com/SheaHawkins/pipecrab/pull/33))
