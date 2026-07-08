# Releasing

Publishing to [crates.io] is automated with [release-plz] via
`.github/workflows/release.yml`.

## How it works

On every push to `main`, release-plz opens (or updates) a **release PR** that
bumps the workspace version and updates the changelogs. Merging that PR:

1. publishes every publishable crate to crates.io, in dependency order, and
2. pushes the git tag(s) and creates the GitHub release(s).

All crates share one version through `version.workspace = true`, so a release
moves the whole set together.

## One-time setup

- **`CARGO_REGISTRY_TOKEN`** — add a crates.io API token with publish rights as
  a repository secret (Settings → Secrets and variables → Actions). release-plz
  reads it to run `cargo publish`.
- **Allow Actions to open PRs** — Settings → Actions → General → *Workflow
  permissions* → enable "Allow GitHub Actions to create and approve pull
  requests" (the release PR is opened with the default `GITHUB_TOKEN`). For CI
  to run on the release PR, use a fine-grained PAT instead; see the release-plz
  docs.

## Published crates

Everything in the workspace is published **except** `pipecrab-test-util` and the
`echo` example, which are marked `publish = false`.

Core:

- `pipecrab-core`, `pipecrab-runtime`, `pipecrab`
- `pipecrab-audio`, `pipecrab-audio-cpal`
- `pipecrab-stt`, `pipecrab-vad`

Model crates (adapt one engine to a capability trait):

- `pipecrab-stt-moonshine` — Moonshine behind the STT `Transcriber` trait
- `pipecrab-vad-silero` — Silero behind the VAD `VoiceActivityDetector` trait

Engines (`{model}-{web|ort}`, pipecrab-free, cfg-selected per target):

- `moonshine-web`, `moonshine-ort` (STT)
- `silero-web`, `silero-ort` (VAD)
- `kokoro-web`, `kokoro-ort` (TTS)

> The model and engine crates are scaffolds today: real types are re-exported
> from the interfaces, but the concrete `Transcriber` / `VoiceActivityDetector`
> impls and inference code land in follow-up work. They are wired into the
> workspace and release graph now so the crate names are reserved and publishing
> is ready.

### Not yet included

The TTS interface and its model crate — `pipecrab-tts` and
`pipecrab-tts-kokoro` — are intentionally deferred. The `kokoro-*` engine crates
above are their eventual backends, reserved ahead of the interface.

## Pre-flight before the first publish

- **Name availability.** The engine crates use generic names (`moonshine-web`,
  `silero-ort`, `kokoro-web`, …). Confirm each is unclaimed on crates.io before
  the first release — a taken name fails `cargo publish`. If one is taken, rename
  the crate (e.g. prefix with `pipecrab-`) in its `Cargo.toml` and the workspace
  `members` + `[workspace.dependencies]` entries.
- **Dry run.** `cargo publish -p <crate> --dry-run` for a spot check, or let the
  release PR's CI exercise the build.

[crates.io]: https://crates.io
[release-plz]: https://release-plz.dev
