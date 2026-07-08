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

[crates.io]: https://crates.io
[release-plz]: https://release-plz.dev
