# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-vad-v0.3.0...pipecrab-vad-v0.4.0) - 2026-07-17

### Other

- update msrv, rust edition ([#44](https://github.com/SheaHawkins/pipecrab/pull/44))

## [0.3.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-vad-v0.2.0...pipecrab-vad-v0.3.0) - 2026-07-13

### Other

- use Arc<f32> for audio engines, prevent copies
- format repo
- group crates into engine/adapters/support with a layering gate ([#37](https://github.com/SheaHawkins/pipecrab/pull/37))

### Changed

- **Breaking:** Pass VAD audio to detectors and scorers as `Arc<[f32]>` so
  worker-backed implementations can retain it without a sample-buffer copy.

## [0.2.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-vad-v0.1.0...pipecrab-vad-v0.2.0) - 2026-07-12

### Added

- *(vad)* [**breaking**] generalize VAD to edge emission; VadStage becomes the gate

### Other

- *(vad)* trim doc justifications and consolidate Debounced tests
- Stt stage, move Speech events to Data lane ([#32](https://github.com/SheaHawkins/pipecrab/pull/32))
- Remove legacy engine crates ([#27](https://github.com/SheaHawkins/pipecrab/pull/27))

## [0.1.0](https://github.com/SheaHawkins/pipecrab/releases/tag/pipecrab-vad-v0.1.0) - 2026-07-08

### Other

- scaffold model + engine crates, wire release-plz ([#20](https://github.com/SheaHawkins/pipecrab/pull/20))
- Add the pipecrab-vad voice-activity-detection interface ([#19](https://github.com/SheaHawkins/pipecrab/pull/19))
