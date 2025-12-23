# `2025-12-23-json-quality` — User Story

- Issue: #63
- Owner: drmoisan
- Status: Draft | In Progress | Complete
- Last Updated: 2025-12-23

## Story Statement

- As a **Developer**, I want **my JSON configuration files to be automatically formatted and validated**, so that **I can avoid syntax errors, duplicates, and inconsistent ordering without manual effort**.
- As a **Repo Maintainer**, I want **CI to enforce JSON schema compliance**, so that **invalid configurations never merge into the main branch**.

## Problem / Why

Config JSON files are inconsistent and hard to review: ordering is random (e.g., tasks.json), duplicates slip through (devcontainer.json), and missing schema validation lets mistakes land in main. We need deterministic formatting and schema-backed validation to keep configs tidy and catch errors early.


## Personas & Scenarios

- Persona: **The Maintainer**
  - **Who**: A senior developer or lead responsible for repo health.
  - **Cares about**: Clean diffs, deterministic builds, and preventing "works on my machine" issues due to bad config.
  - **Frustrations**: Reviewing PRs with massive diffs just because keys were reordered; debugging CI failures caused by a trailing comma in a strict JSON file.
  - **Goal**: "Set it and forget it" tooling that enforces consistency.

- Scenario: **Adding a new VS Code Task**
  - **Who**: A developer adding a build task.
  - **Trigger**: Need to add a new `run-build` task to `.vscode/tasks.json`.
  - **Action**: They copy-paste an existing task to the bottom of the file, leaving keys in random order and maybe a duplicate key.
  - **Step**: They run the `Fix All` task (or pre-commit hook).
  - **Outcome**: The system detects the file is governed. It runs `jq` to sort keys and fix indentation. It then runs `jsonschema` validation.
  - **Result**: The duplicate key is removed (or flagged if invalid JSON), keys are sorted alphabetically. The developer commits a clean, compliant file.


## Acceptance Criteria

- [ ] Governed config paths are documented (e.g., .vscode/**/*.json, .devcontainer/*.json, scripts/**/config*.json, docs/examples/**/*.json) with explicit excludes for data/artifacts.
- [ ] Formatter step sorts keys deterministically (except opt-out list) and leaves arrays untouched; it fails loudly on JSONC/trailing commas unless explicitly supported.
- [ ] Validator step enforces `$schema` for governed files and reports file + JSON path on failure.
- [ ] Fix All/Run All Checks include JSON format then validate before Python/PowerShell checks; CI job enforces the same.
- [ ] README/developer-tooling documents usage, globs, opt-outs, and schema cache behavior.


## Non-Goals

- **Data Formatting**: We will not format large data files (e.g., `data/corpus/**/*.json`) to avoid performance hits and massive diffs.
- **Comment Preservation**: Since we are using `jq`, we will not support JSONC (comments) in governed files. Files with comments must either be converted to strict JSON or excluded from this toolchain.
