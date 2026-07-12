

## You own the changes you make.
AI use in this project is tolerated, but architectural decisions have rippling effects and implementations need to be sound. This project does not tolerate "AI Slop", either from the reporter or contributor's perspective. 

AI should be used for finite, well-reasoned changes. PRs need to be compartmental and singular in purpose. The entire change being made should be understood by _you_ and the code is made under _your_ name, not a bot's. Benchmarks should be included when they touch sensitive components.

## Commit and PR conventions
This project follows [conventionalcommits.org](https://www.conventionalcommits.org); the allowed types are listed in `AGENTS.md`. CI enforces it on every pull request (`.github/workflows/conventional-commits.yml`): the **PR title** is linted (it becomes the commit on a squash merge) and every **commit message** in the PR is checked with [cocogitto](https://cocogitto.io) (`cog.toml`). Both must pass to merge.
