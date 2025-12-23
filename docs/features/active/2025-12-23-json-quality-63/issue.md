# json-quality (Issue #63)

- Date captured: 2025-12-23
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/json-quality/ (Issue #63)

- Issue: #63
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/63
- Last Updated: 2025-12-23
## Problem / Why
Config JSON files are inconsistent and hard to review: ordering is random (e.g., tasks.json), duplicates slip through (devcontainer.json), and missing schema validation lets mistakes land in main. We need deterministic formatting and schema-backed validation to keep configs tidy and catch errors early.

## Proposed Behavior
Add a JSON quality toolchain for config files (not data) that:

- Formats governed JSON with jq using sorted keys by default, preserving arrays and failing fast on JSONC/trailing commas.
- Validates governed files against their `$schema` (e.g., devcontainer, VS Code tasks/settings/extensions) via jsonschema with clear error paths.
- Runs from VS Code tasks, the Fix All/Run All Checks pipeline, and CI, with clear include/exclude globs for config vs. data.

## Acceptance Criteria (early draft)
- [ ] Governed config paths are documented (e.g., .vscode/**/*.json, .devcontainer/*.json, scripts/**/config*.json, docs/examples/**/*.json) with explicit excludes for data/artifacts.
- [ ] Formatter step sorts keys deterministically (except opt-out list) and leaves arrays untouched; it fails loudly on JSONC/trailing commas unless explicitly supported.
- [ ] Validator step enforces `$schema` for governed files and reports file + JSON path on failure.
- [ ] Fix All/Run All Checks include JSON format then validate before Python/PowerShell checks; CI job enforces the same.
- [ ] README/developer-tooling documents usage, globs, opt-outs, and schema cache behavior.

## Constraints & Risks
- Sorting keys could conflict with tools or consumers that rely on insertion order; maintain an allowlist to skip sorting where needed.
- Strict JSON (no comments/trailing commas) may require cleaning existing files; if JSONC is needed, choose a JSON5-capable formatter instead of jq.
- Schema fetches rely on network; add caching/fallback to reduce flakiness.
- Large or generated JSON must stay excluded to avoid noisy diffs and long runs.

## Test Conditions to Consider
- [ ] Unit tests for formatter behavior: sorted keys, preserved arrays, failure on JSONC/trailing commas, opt-out handling.
- [ ] Unit tests for validator: missing `$schema`, schema fetch failure, validation error paths, cache usage.
- [ ] Integration tests for Fix All path ordering (format → validate → Ruff/Pyright/Pytest) and VS Code tasks.
- [ ] Sample governed files validated in CI to prove end-to-end wiring.

## Next Step
- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/json-quality/` folder from the template
