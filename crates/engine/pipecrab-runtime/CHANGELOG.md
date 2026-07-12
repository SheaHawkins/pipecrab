# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/SheaHawkins/pipecrab/compare/pipecrab-runtime-v0.1.0...pipecrab-runtime-v0.2.0) - 2026-07-12

### Other

- Partial Transcripts ([#29](https://github.com/SheaHawkins/pipecrab/pull/29))
- release v0.1.0 ([#21](https://github.com/SheaHawkins/pipecrab/pull/21))

## [0.1.0](https://github.com/SheaHawkins/pipecrab/releases/tag/pipecrab-runtime-v0.1.0) - 2026-07-08

### Other

- scaffold model + engine crates, wire release-plz ([#20](https://github.com/SheaHawkins/pipecrab/pull/20))
- Add the pipecrab-vad voice-activity-detection interface ([#19](https://github.com/SheaHawkins/pipecrab/pull/19))
- Propogate MaybeSend trait into pipecrab-audio, sugar ([#17](https://github.com/SheaHawkins/pipecrab/pull/17))
- Blanket MaybeSendSync across wasm/native architectures ([#16](https://github.com/SheaHawkins/pipecrab/pull/16))
- Add pipeline-audio, AudioChunks, AudioSink ([#12](https://github.com/SheaHawkins/pipecrab/pull/12))
- Add a helper for offloading work to a thread ([#11](https://github.com/SheaHawkins/pipecrab/pull/11))
- Stages, Pipelines, PipelineBuilder ([#10](https://github.com/SheaHawkins/pipecrab/pull/10))
- Wasm Compilation ([#9](https://github.com/SheaHawkins/pipecrab/pull/9))
- Frame flushing & survivor frames ([#8](https://github.com/SheaHawkins/pipecrab/pull/8))
- Add outbound channels ([#6](https://github.com/SheaHawkins/pipecrab/pull/6))
- Split System and Data frame dichotomy ([#5](https://github.com/SheaHawkins/pipecrab/pull/5))
- Remove direction from data lane ([#4](https://github.com/SheaHawkins/pipecrab/pull/4))
- Add fatal flag to errors ([#3](https://github.com/SheaHawkins/pipecrab/pull/3))
- Allow up to 1 tokio alloc
- Allow select! to alloc
- Inbound trait, hoist test utils, better containerization
