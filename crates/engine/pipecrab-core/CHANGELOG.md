# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-core-v0.2.0...pipecrab-core-v0.3.0) - 2026-07-13

### Other

- format repo
- group crates into engine/adapters/support with a layering gate ([#37](https://github.com/SheaHawkins/pipecrab/pull/37))

## [0.2.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-core-v0.1.0...pipecrab-core-v0.2.0) - 2026-07-12

### Added

- *(vad)* [**breaking**] generalize VAD to edge emission; VadStage becomes the gate

### Other

- *(vad)* trim doc justifications and consolidate Debounced tests
- Stt stage, move Speech events to Data lane ([#32](https://github.com/SheaHawkins/pipecrab/pull/32))
- Partial Transcripts ([#29](https://github.com/SheaHawkins/pipecrab/pull/29))
- release v0.1.0 ([#21](https://github.com/SheaHawkins/pipecrab/pull/21))

## [0.1.0](https://github.com/SheaHawkins/pipecrab/releases/tag/pipecrab-core-v0.1.0) - 2026-07-08

### Other

- scaffold model + engine crates, wire release-plz ([#20](https://github.com/SheaHawkins/pipecrab/pull/20))
- Add the pipecrab-vad voice-activity-detection interface ([#19](https://github.com/SheaHawkins/pipecrab/pull/19))
- Add pipeline-audio, AudioChunks, AudioSink ([#12](https://github.com/SheaHawkins/pipecrab/pull/12))
- Frame flushing & survivor frames ([#8](https://github.com/SheaHawkins/pipecrab/pull/8))
- Frame dropping/forwarding decisions are explicit ([#7](https://github.com/SheaHawkins/pipecrab/pull/7))
- Split System and Data frame dichotomy ([#5](https://github.com/SheaHawkins/pipecrab/pull/5))
- Add fatal flag to errors ([#3](https://github.com/SheaHawkins/pipecrab/pull/3))
- Inbound trait, hoist test utils, better containerization
- Add pipecrab core, alloc tests, ci
