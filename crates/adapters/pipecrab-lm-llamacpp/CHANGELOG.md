# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Track `pipecrab-lm`'s structured interface: `generate` accepts the effective
  `&[ToolDefinition]` (ignored — no native tool parsing) and streams decoded text
  as `ModelDelta::Text`. History renders the structured `Message` enum per role.

## [0.4.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-lm-llamacpp-v0.3.0...pipecrab-lm-llamacpp-v0.4.0) - 2026-07-17

### Other

- release v0.4.0
