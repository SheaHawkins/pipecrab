## You own the changes you make.
AI use in this project is tolerated, but architectural decisions have rippling effects and implementations need to be sound. This project does not tolerate "AI Slop", either from the reporter or contributor's perspective. 

AI should be used for finite, well-reasoned changes. PRs need to be compartmental and singular in purpose. The entire change being made should be understood by _you_, not a bot. Benchmarks should be included when they touch sensitive components. 

## Commit and PR conventions
We follow conventionalcommits.org. The essential rules are attached below. Follow the same conventions for PR names:

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
