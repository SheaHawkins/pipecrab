# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add a streaming `ResamplerStage` for sample-rate and channel-count conversion.

## [0.1.0](https://github.com/SheaHawkins/pipecrab/releases/tag/pipecrab-audio-v0.1.0) - 2026-07-08

### Other

- scaffold model + engine crates, wire release-plz ([#20](https://github.com/SheaHawkins/pipecrab/pull/20))
- Add the pipecrab-vad voice-activity-detection interface ([#19](https://github.com/SheaHawkins/pipecrab/pull/19))
- Propogate MaybeSend trait into pipecrab-audio, sugar ([#17](https://github.com/SheaHawkins/pipecrab/pull/17))
- Blanket MaybeSendSync across wasm/native architectures ([#16](https://github.com/SheaHawkins/pipecrab/pull/16))
- Add a cpal source+sink and demo app ([#13](https://github.com/SheaHawkins/pipecrab/pull/13))
- Add pipeline-audio, AudioChunks, AudioSink ([#12](https://github.com/SheaHawkins/pipecrab/pull/12))
