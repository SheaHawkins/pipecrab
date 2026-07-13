# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-stt-v0.2.0...pipecrab-stt-v0.3.0) - 2026-07-13

### Other

- use Arc<f32> for audio engines, prevent copies
- format repo
- group crates into engine/adapters/support with a layering gate ([#37](https://github.com/SheaHawkins/pipecrab/pull/37))

### Changed

- **Breaking:** Pass streaming and one-shot STT audio as `Arc<[f32]>` so
  worker-backed implementations can enqueue it without a sample-buffer copy.

## [0.2.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-stt-v0.1.0...pipecrab-stt-v0.2.0) - 2026-07-12

### Other

- *(stt)* add a test for rejecting double begins
- *(stt)* Remove in_speech state, ringbuffer from stt
- Stt stage, move Speech events to Data lane ([#32](https://github.com/SheaHawkins/pipecrab/pull/32))
- Streaming transcriber ([#30](https://github.com/SheaHawkins/pipecrab/pull/30))
- Partial Transcripts ([#29](https://github.com/SheaHawkins/pipecrab/pull/29))
- Remove legacy engine crates ([#27](https://github.com/SheaHawkins/pipecrab/pull/27))

## [0.1.0](https://github.com/SheaHawkins/pipecrab/releases/tag/pipecrab-stt-v0.1.0) - 2026-07-08

### Other

- scaffold model + engine crates, wire release-plz ([#20](https://github.com/SheaHawkins/pipecrab/pull/20))
- Add the pipecrab-vad voice-activity-detection interface ([#19](https://github.com/SheaHawkins/pipecrab/pull/19))
- Blanket MaybeSendSync across wasm/native architectures ([#16](https://github.com/SheaHawkins/pipecrab/pull/16))
- Add interface for stt stages ([#14](https://github.com/SheaHawkins/pipecrab/pull/14))
