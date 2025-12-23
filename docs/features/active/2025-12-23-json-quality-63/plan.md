# 2025-12-23-json-quality - Plan

- Issue: #63
- Owner: drmoisan
- Last Updated: 2025-12-23

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- Python Code Change Policy: [`.github/instructions/python-code-change.instructions.md`](../../../../.github/instructions/python-code-change.instructions.md)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

### Phase 0: Context & Inputs
- [x] [P0-T1] Read `.github/instructions/general-code-change.instructions.md` and `.github/instructions/python-code-change.instructions.md` to establish baseline rules.
- [x] [P0-T2] Read `.github/instructions/general-unit-test.instructions.md` and `.github/instructions/python-unit-test.instructions.md` to confirm test expectations.
- [x] [P0-T3] Verify `jq` is installed in the environment (`jq --version`) and note if it needs to be added to the devcontainer.
- [x] [P0-T4] Verify `jsonschema` is installed (`pip show jsonschema`) and note if it needs to be added to dependencies.

### Phase 1: Dependencies & Environment
- [x] [P1-T1] Add `jsonschema` to `pyproject.toml` dev dependencies using `poetry add -D jsonschema`.
- [x] [P1-T2] Update `.devcontainer/devcontainer.json` (features or postCreateCommand) to ensure `jq` is installed.

### Phase 2: Shared Configuration
- [x] [P2-T1] Create `scripts/dev_tools/json_config.py` defining `GOVERNED_GLOBS` (e.g., `.vscode/*.json`, `.devcontainer/*.json`) and `EXCLUDE_GLOBS`.

### Phase 3: Formatter Implementation
- [x] [P3-T1] Implement `scripts/dev_tools/format_json.py` to find files using `json_config.py` and run `jq --sort-keys` (using temp files for safety).
- [x] [P3-T2] Create `tests/scripts/dev_tools/test_format_json.py` with test cases for sorting, exclusion, and invalid JSON handling.

### Phase 4: Validator Implementation
- [x] [P4-T1] Implement `scripts/dev_tools/validate_json.py` to fetch schemas (from `$schema` key) and validate files using `jsonschema`.
- [x] [P4-T2] Create `tests/scripts/dev_tools/test_validate_json.py` with test cases for schema fetching, caching, and validation errors.

### Phase 5: Config Cleanup & Schema Association
*Note: These tasks must be done sequentially or carefully to avoid breaking the environment before the toolchain is ready.*
- [x] [P5-T1] Remove comments and trailing commas from `.devcontainer/devcontainer.json` to make it strict JSON compatible.
- [x] [P5-T2] Add `$schema` property to `.devcontainer/devcontainer.json` pointing to the official schema.
- [x] [P5-T3] Remove comments and trailing commas from `.vscode/tasks.json` to make it strict JSON compatible.
- [x] [P5-T4] Add `$schema` property to `.vscode/tasks.json` pointing to the official schema.

### Phase 6: Integration
- [x] [P6-T1] Update `scripts/dev_tools/fix_all.py` to invoke `format_json` and `validate_json` in the correct order.
- [x] [P6-T2] Update `.vscode/tasks.json` to add `JSON: format` and `JSON: validate` tasks.
- [x] [P6-T3] Update `.vscode/tasks.json` task `QC: 5 Run All Checks` to include the new JSON tasks.

### Phase 7: Documentation
- [x] [P7-T1] Update `docs/developer-tooling.md` to describe the new JSON toolchain, including how to run it and how to configure it.

## Test Plan

- **Unit**:
  - `test_format_json.py`: Verify `jq` invocation, file discovery, and error handling.
  - `test_validate_json.py`: Verify schema fetching, caching logic, and validation reporting.
- **Integration**:
  - Run `fix_all` and verify it formats and validates JSON files.
  - Verify VS Code tasks `JSON: format` and `JSON: validate` execute successfully.
- **Manual**:
  - Introduce a syntax error in `tasks.json` and verify `validate_json` catches it.
  - Introduce a key ordering change in `tasks.json` and verify `format_json` fixes it.

## Open Questions / Notes

- **JSONC Support**: We are explicitly NOT supporting JSONC (comments) to keep the toolchain simple (`jq` + `jsonschema`). Config files with comments must be converted to strict JSON.
- **Schema Caching**: `validate_json.py` should implement basic caching to avoid network dependency on every run.
