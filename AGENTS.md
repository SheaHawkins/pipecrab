1. Refer to ARCHITECTURE.md for basic rules and decisions about this repo.
2. wasm32 check is per-crate, NOT `--workspace`. The browser-portability gate covers only the platform-neutral crates — `pipecrab-core`, `pipecrab-runtime`, `pipecrab-audio`, `pipecrab-stt`, `pipecrab-vad` — each checked individually with `cargo check -p <crate> --target wasm32-unknown-unknown`. Native backend crates are exempt and WILL fail a wasm build: `pipecrab-audio-cpal` is native-only (the browser audio path will be a separate crate). So do not run `cargo check --workspace --target wasm32-unknown-unknown` — a cpal failure there is expected, not a regression. `.github/workflows/ci.yml` is the source of truth for which crates are gated.
3. We follow conventionalcommits.org. The essential rules are attached:

– Commits MUST be prefixed with a type, which consists of a noun, feat, fix, etc., followed by the OPTIONAL scope, OPTIONAL !, and REQUIRED terminal colon and space.
- The type feat MUST be used when a commit adds a new feature to your application or library.
- The type fix MUST be used when a commit represents a bug fix for your application.
- A scope MAY be provided after a type. A scope MUST consist of a noun describing a section of the codebase surrounded by parenthesis, e.g., fix(parser):
- A description MUST immediately follow the colon and space after the type/scope prefix. The description is a short summary of the code changes, e.g., fix: array parsing issue when multiple spaces were contained in string.
- A longer commit body MAY be provided after the short description, providing additional contextual information about the code changes. The body MUST begin one blank line after the description.
- A commit body is free-form and MAY consist of any number of newline separated paragraphs.

A list of allowed types:
  'build',
  'chore',
  'ci',
  'docs',
  'feat',
  'fix',
  'perf',
  'refactor',
  'revert',
  'style',
  'test'
