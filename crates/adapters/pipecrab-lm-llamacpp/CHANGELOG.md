# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-lm-llamacpp-v0.5.1...pipecrab-lm-llamacpp-v0.6.0) - 2026-07-28

### Added

- hermes-voice-agent example ([#72](https://github.com/SheaHawkins/pipecrab/pull/72))
- *(llamacpp)* parse and emit tool calls ([#71](https://github.com/SheaHawkins/pipecrab/pull/71))

### Added

- `LlamaCppConfig::with_assistant_prefix`: text prefilled after the generation
  prompt, so the model continues from it and none of it reaches the delta
  stream. `llama_chat_apply_template` takes only role/content pairs, so a
  template kwarg cannot be passed — this applies one by hand. Qwen 3's
  non-thinking mode is the motivating case: without it a Qwen 3 GGUF reasons
  into the transcript, which a voice pipeline speaks aloud.

- Emit `ModelDelta::ToolCall` from the native adapter. `generate` now honours the
  `&[ToolDefinition]` it is handed: a `ToolDialect` renders declarations into the
  system message, converts the tool schemas into a GBNF attached as a *lazy*
  grammar triggered by the call's open delimiter, and reads the captured body
  back as a name and arguments. `ChatMlXml` implements the Qwen 2.5/3 and Nous
  Hermes convention and is the default; `LlamaCppConfig::with_tool_dialect`
  replaces it. A defaulted dialect is checked against the GGUF's `general.name`
  before the first tool-carrying generation.
- Call text is withheld from the delta stream as it is captured, including a
  trigger split across token boundaries, so no fragment of the call syntax
  reaches a transcript and gets spoken. A body left unterminated by `max_tokens`
  yields `LmError::IncompleteToolCall` rather than a partial call.

### Fixed

- Drop the redundant `sampler.accept` after `sampler.sample`, which already
  accepts the sampled token. The double accept advanced a grammar sampler twice
  per token and aborted the process.

### Changed

- Tool results render through the dialect into a role the chat template knows
  (`user`, wrapped in `<tool_response>`), and an assistant turn's `tool_calls`
  render back into its content. Passing no tools leaves both, the prompt, and
  the delta stream exactly as they were.

## [0.5.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-lm-llamacpp-v0.4.0...pipecrab-lm-llamacpp-v0.5.0) - 2026-07-22

### Added

- *(lm)* add tool definitions, model deltas with tool calls ([#63](https://github.com/SheaHawkins/pipecrab/pull/63))

### Other

- release v0.4.0

### Changed

- Track `pipecrab-lm`'s structured interface: `generate` accepts the effective
  `&[ToolDefinition]` (ignored — no native tool parsing) and streams decoded text
  as `ModelDelta::Text`. History renders the structured `Message` enum per role.

## [0.4.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-lm-llamacpp-v0.3.0...pipecrab-lm-llamacpp-v0.4.0) - 2026-07-17

### Other

- release v0.4.0
