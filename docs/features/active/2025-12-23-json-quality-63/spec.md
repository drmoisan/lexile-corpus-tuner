# 2025-12-23-json-quality — Spec

- Issue: #63
- Owner: drmoisan
- Last Updated: 2025-12-23

## Overview

Config JSON files are inconsistent and hard to review: ordering is random (e.g., tasks.json), duplicates slip through (devcontainer.json), and missing schema validation lets mistakes land in main. We need deterministic formatting and schema-backed validation to keep configs tidy and catch errors early.


## Behavior

Add a JSON quality toolchain for config files (not data) that:

- Formats governed JSON with jq using sorted keys by default, preserving arrays and failing fast on JSONC/trailing commas.
- Validates governed files against their `$schema` (e.g., devcontainer, VS Code tasks/settings/extensions) via jsonschema with clear error paths.
- Runs from VS Code tasks, the Fix All/Run All Checks pipeline, and CI, with clear include/exclude globs for config vs. data.


## Inputs / Outputs

- **Inputs**:
  - **Target Files**: Defined by globs (e.g., `.vscode/*.json`) and excludes (e.g., `data/**`) in the script configuration.
  - **Schemas**: `$schema` property in JSON files or hardcoded mapping for known files (e.g., `tasks.json`).
  - **CLI Flags**: `--check` (verify without modifying), `--verbose` (debug logs).

- **Outputs**:
  - **Formatted Files**: In-place updates (sorted keys, 2-space indent) using `jq`.
  - **Validation Report**: Console output listing validation errors (File, Path, Message).
  - **Exit Code**: Non-zero if validation fails or if formatting would change files (in `--check` mode).

## API / CLI Surface

**Scripts**:
- `python -m scripts.dev_tools.format_json [--check] [paths...]`
  - Wraps `jq --sort-keys`.
  - Returns 0 on success, 1 on error/diff (in check mode).
- `python -m scripts.dev_tools.validate_json [paths...]`
  - Uses `jsonschema` to validate against `$schema`.
  - Returns 0 on success, 1 on validation failure.

**VS Code Tasks**:
- `JSON: format`: Runs formatter on governed files.
- `JSON: validate`: Runs validator on governed files.
- `Fix All`: Updated to include `JSON: format` -> `JSON: validate`.

## Data & State

- **Schema Cache**: Optional local cache (e.g., `.cache/schemas`) to prevent network timeouts during validation.
- **File State**: Files are modified in-place. No database or external state.

## Constraints & Risks

- Sorting keys could conflict with tools or consumers that rely on insertion order; maintain an allowlist to skip sorting where needed.
- Strict JSON (no comments/trailing commas) may require cleaning existing files; if JSONC is needed, choose a JSON5-capable formatter instead of jq.
- Schema fetches rely on network; add caching/fallback to reduce flakiness.
- Large or generated JSON must stay excluded to avoid noisy diffs and long runs.


## Definition of Done

- [ ] Behavior matches acceptance criteria
- [ ] Tests updated/added
- [ ] Docs updated (README, docs/features/active/... links)
- [ ] Telemetry/logging (if applicable)

## Seeded Test Conditions (from potential)
- [ ] Unit tests for formatter behavior: sorted keys, preserved arrays, failure on JSONC/trailing commas, opt-out handling.
- [ ] Unit tests for validator: missing `$schema`, schema fetch failure, validation error paths, cache usage.
- [ ] Integration tests for Fix All path ordering (format → validate → Ruff/Pyright/Pytest) and VS Code tasks.
- [ ] Sample governed files validated in CI to prove end-to-end wiring.
