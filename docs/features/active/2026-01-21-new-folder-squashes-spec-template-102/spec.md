# 2026-01-21-new-folder-squashes-spec-template (Spec)

- **Issue:** #102
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-01-21T10-41
- **Status:** Draft
- **Version:** 0.1

## Context
Creating an active bug folder via `Dev: 3 Create Active Folder` overwrites the bug `spec.md` template’s intent by pre-populating the `## Proposed Fix` (and `## Test Strategy`) sections using the early “Proposed Fix / Validation Ideas” content from the potential bug file that becomes `issue.md`.

This prematurely locks the design direction and loses the intended separation where early hypotheses belong in `issue.md`, and the final `spec.md` is filled only after research is complete.

Environment:
- OS/version: Linux (devcontainer: Debian GNU/Linux 12 (bookworm))
- Python version: Poetry-managed; repo CI targets Python 3.10–3.13 (exact runtime version TBD)
- Command/flags used:
	- VS Code task: `Dev: 3 Create Active Folder`
	- Under the hood:
		- `poetry run python -m scripts.dev_tools.new_active_feature_folder --feature-name <bug-name> --type bug --issue-number <issue|auto>`
- Data source or fixture:
	- Potential bug markdown file under `docs/features/potential/` containing early “Proposed Fix / Validation Ideas” content
	- Bug `spec.md` template under `docs/features/templates/bug/spec.md`

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Create a potential bug file under `docs/features/potential/` that includes an early hypothesis in `## Proposed Fix / Validation Ideas`.
2. Run the VS Code task `Dev: 3 Create Active Folder` with:
	- `--type bug`
	- `--feature-name <bug-name>`
	- `--issue-number <issue|auto>`
3. Open the generated active folder under `docs/features/active/<slug>/`.
4. Inspect `docs/features/active/<slug>/spec.md`.

Expected:
The created `docs/features/active/<slug>/spec.md` should preserve the robust bug `spec.md` template structure and leave `## Proposed Fix` (and other design-commitment sections) unpopulated.

Early proposed fixes/hypotheses should remain only in the moved `issue.md` (the promoted potential file), and research should drive what ultimately gets written into `spec.md`.

Actual:
`docs/features/active/<slug>/spec.md` has its `## Proposed Fix` (and `## Test Strategy`) sections pre-populated from the potential bug file’s `## Proposed Fix / Validation Ideas` section.

This effectively “squashes” (overwrites the intent of) the `spec.md` template by transferring early thinking directly into the spec, narrowing the later research space.

Logs / Screenshots:
- [ ] Attached minimal logs or screenshot
- Snippet:

	The active-folder creation script currently injects the potential bug’s “Proposed Fix / Validation Ideas” into the bug spec:

	- `scripts/dev_tools/new_active_feature_folder.py` (bug doc seeding flow)
		- When `bug_validation` content exists, it is applied to both `Proposed Fix` and `Test Strategy` in `spec.md`.


## Scope & Non-Goals
- In scope:
	- Update bug-folder doc seeding so the generated `spec.md` preserves the bug template prompts and does not receive early-hypothesis content from the promoted potential file.
	- Keep automatic seeding limited to factual, non-design-commitment sections:
		- `## Context`
		- `## Repro & Evidence`
		- `## Root Cause Analysis`
	- Preserve the existing behavior that moves the promoted potential file into the active folder as `issue.md`.
- Out of scope / non-goals:
	- Changing feature/refactor/epic folder generation behavior.
	- Changing the bug `plan.md` template or plan-file timestamping behavior.
	- Adding new CLI flags, configuration schemas, or any runtime pipeline behavior.
- Explicitly excluded systems, integrations, or datasets:
	- Any external services (GitHub API aside from optional metadata fetch), corpora, or model pipelines.

## Root Cause Analysis
The seeding logic for bug folders copies sections out of the promoted potential bug file and applies them directly into the bug `spec.md`.

Specifically, the bug flow treats the potential bug’s `## Proposed Fix / Validation Ideas` as `bug_validation` and then writes that body into `spec.md` under both:
- `## Proposed Fix`
- `## Test Strategy`

Relevant implementation entry point:
- `scripts/dev_tools/new_active_feature_folder.py` (called by the VS Code task `Dev: 3 Create Active Folder`)


## Proposed Fix

### Design summary (what changes where):
Stop treating the promoted potential bug section `## Proposed Fix / Validation Ideas` as seed material for `spec.md`.

Concretely, in `scripts/dev_tools/new_active_feature_folder.py`, update the bug-folder path inside `update_feature_docs(...)` so that `bug_validation` (extracted via `get_section(potential_content, "Proposed Fix / Validation Ideas")`) is not inserted into any section of `spec.md`.

Instead:
- Keep early hypotheses/validation ideas in `docs/features/active/<slug>/issue.md` (the promoted potential file that `create_active_folder(...)` moves into the active folder).
- Keep automatic seeding limited to the factual sections (`Context`, `Repro & Evidence`, `Root Cause Analysis`).

### Boundaries and invariants to preserve:
- The bug `spec.md` template structure must remain intact (including all `##` and `###` subsections under `## Proposed Fix`).
- The promoted potential file must still be moved into the active folder as `issue.md` (so early hypotheses are preserved).
- No changes to how `issue_number` is resolved (explicit `--issue-number`, `auto`, or parsed from the potential file).
- No changes to plan file creation and timestamp behavior.

### Dependencies or blocked work:
- None.

### Implementation strategy (what changes, not sequencing):

Decision logic:
- Treat `bug_validation` as *source-only* content (kept in `issue.md`) rather than as content to be merged into `spec.md`.
	
#### Files/modules to change:
- `scripts/dev_tools/new_active_feature_folder.py`
	- In the bug branch of `update_feature_docs(...)`, remove (or gate off) the `_update_section_body(..., "Test Strategy", ...)` call that prepends `bug_validation`.
	- Ensure no other bug-path code writes `bug_validation` into any `spec.md` section.
- Unit tests under `tests/scripts/dev_tools/`
	- Update/add tests to assert that `spec.md` does not contain `bug_validation` content, while `issue.md` still does.

#### Functions/classes/CLI commands impacted:
- `scripts.dev_tools.new_active_feature_folder.update_feature_docs(...)` (bug folder doc seeding)
- `scripts.dev_tools.new_active_feature_folder.create_active_folder(...)` (invariant: still moves the potential file into `issue.md`)
- Helpers used in the bug path:
	- `get_section(...)` (extracts potential sections, including `Proposed Fix / Validation Ideas`)
	- `_update_section_body(...)` / `_prepend_to_section_body(...)` (should no longer be invoked for `Test Strategy` seeding)

#### Data flow and validation changes:
- Before: potential `## Proposed Fix / Validation Ideas` -> `bug_validation` -> injected into `spec.md`.
- After: potential `## Proposed Fix / Validation Ideas` remains only in `issue.md`; `spec.md` receives only safe factual sections.

#### Error handling and logging updates:
- No behavior changes required.
- Keep existing error handling for missing template folders and existing-target conflicts.

#### Rollback/feature-flag considerations (if applicable):
- Not applicable (developer tooling only). Roll back by reverting the generator change.

### Technical specifications (interfaces/contracts):

Contract:
- When `create_active_folder(..., feature_type="bug")` is invoked and a potential file exists:
	- The potential file is moved into the new active folder as `issue.md`.
	- `spec.md` is created from `docs/features/templates/bug/spec.md` with header placeholders filled.
	- Only `Context`, `Repro & Evidence`, and `Root Cause Analysis` may be auto-seeded from the potential file.
	- `Proposed Fix` and `Test Strategy` must remain in their template/prompt form (no injection of potential content).

#### Inputs/outputs and formats:
- Inputs:
	- Potential bug markdown file under `docs/features/potential/` with section headings (e.g., `## Summary`, `## Steps to Reproduce`, `## Proposed Fix / Validation Ideas`).
	- Template file: `docs/features/templates/bug/spec.md`.
- Outputs:
	- `docs/features/active/<slug>/spec.md` (template structure preserved; no validation-idea injection).
	- `docs/features/active/<slug>/issue.md` (verbatim promoted potential content).

#### Required configuration keys and defaults:
- None.

#### Backward-compatibility expectations:
- This is a developer-tooling behavior change only.
- Previously-created active bug folders are not migrated.
- Existing CLI surface remains unchanged.

#### Performance constraints (latency/throughput/memory):
- Negligible; this is small-file markdown manipulation.

## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
	- The workspace follows the standard doc layout under `docs/features/{templates,potential,active}`.
	- Potential bug files use the section headings expected by `get_section(...)`.
- Constraints (budget, performance, compatibility):
	- Unit tests must not create temporary files (repo policy). Use an in-memory filesystem pattern.
	- Must pass the Python toolchain (Black → Ruff → Pyright → Pytest).
- External dependencies (services, libraries, releases):
	- None.

## Data / API / Config Impact
- User-facing or API changes:
	- None.
- Data or migration considerations:
	- None.
- Logging/telemetry updates (if any):
	- None.
- Compatibility notes (CLI flags, config schemas, versioning):
	- No CLI/config changes.

## Test Strategy
Seeded from issue:

- [ ] Unit coverage areas
	- Update/add tests in `tests/scripts/dev_tools/test_new_active_feature_folder.py` to assert that, for bug folders, `spec.md` does **not** receive the “Proposed Fix / Validation Ideas” content from the potential file.
	- Keep the existing behavior that moves the potential file into the active folder as `issue.md` (so early hypotheses are preserved there).
- [ ] Integration scenario to retest
	- Create a potential bug file with non-trivial “Proposed Fix / Validation Ideas”, run `Dev: 3 Create Active Folder` with `--type bug`, and verify the resulting `spec.md` retains only template structure (no pre-populated Proposed Fix/Test Strategy content).
- [ ] Manual verification notes
	- Confirm the resulting `issue.md` still contains the early proposed fix content verbatim.
	- Confirm `spec.md` still contains the required headings but with empty/placeholder bodies in `Proposed Fix`/`Test Strategy`.

Additional unit coverage areas (must be deterministic and filesystem-free):
- Assert that the text under the potential file’s `## Proposed Fix / Validation Ideas` does not appear anywhere in the generated `spec.md`.
- Assert that the same text still appears in the generated `issue.md`.
- Assert that `spec.md` still contains the full bug template `## Proposed Fix` subsection headings (e.g., `### Design summary (what changes where):`).

- Regression tests to add or update:
- Update:
	- `tests/scripts/dev_tools/test_new_active_feature_folder_bug_template_preserved.py::test_bug_folder_preserves_proposed_fix_template_subsections` to assert that validation text is not present in `spec.md` at all, and is present in `issue.md`.
- Unit tests (pytest) for the fixed behavior and boundaries:
- Add a focused unit test in `tests/scripts/dev_tools/test_new_active_feature_folder.py` for bug folders verifying `Test Strategy` remains unseeded even when `bug_validation` exists.
- Edge cases and negative scenarios (invalid inputs, missing data, boundary values):
- No potential file found: generator still creates the folder and leaves design sections unseeded.
- Potential file exists but lacks `## Proposed Fix / Validation Ideas`: generator leaves `spec.md` unseeded.
- Error handling and logging verification:
- Confirm existing `FileExistsError` and `FileNotFoundError` behaviors remain unchanged.
- Coverage impact and targets for changed lines/modules:
- Maintain coverage for the changed bug-path lines in `scripts/dev_tools/new_active_feature_folder.py`.
- Toolchain commands to run (format → lint → type-check → test):
- `poetry run black .`
- `poetry run ruff check`
- `poetry run pyright`
- `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
- Manual validation steps (if required):
- Create a potential bug file containing non-trivial `## Proposed Fix / Validation Ideas`, run `Dev: 3 Create Active Folder` with `--type bug`, and verify:
	- `spec.md` contains the template headings with no injected validation text.
	- `issue.md` contains the validation text verbatim.


## Acceptance Criteria
- [ ] Repro steps now produce the expected behavior in all documented environments (creating a bug folder no longer injects “Proposed Fix / Validation Ideas” into `spec.md`).
- [ ] Regression test(s) added and passing:
	- `tests/scripts/dev_tools/test_new_active_feature_folder_bug_template_preserved.py::test_bug_folder_preserves_proposed_fix_template_subsections` (updated for this behavior)
	- New/updated test in `tests/scripts/dev_tools/test_new_active_feature_folder.py` covering the bug flow’s `Test Strategy` behavior (nodeid to be finalized when added)
- [ ] Edge cases and invalid inputs are handled with correct errors or fallbacks:
	- No potential file found
	- Potential file exists but lacks `## Proposed Fix / Validation Ideas`
- [ ] No unintended behavior changes outside the defined scope (feature/refactor/epic folder generation unchanged).
- [ ] Required logs/telemetry updated and validated (if applicable) (N/A for this tooling-only change).
- [ ] Performance constraints met or explicitly waived with rationale (N/A; markdown manipulation only).
- [ ] Full toolchain pass completed (format → lint → type-check → test).
- [ ] Docs/config references updated to match the new behavior (N/A unless developer docs are updated to explicitly call out the `issue.md` vs `spec.md` separation).

## Risks & Mitigations
- Technical or operational risks:
	- Removing auto-seeded validation ideas may reduce convenience for users who relied on the generator to prefill `Test Strategy`.
	- Future template edits could reintroduce accidental seeding behavior if regression tests are not kept aligned with the template.
- Mitigations and rollbacks:
	- Mitigation: ensure the promoted potential file is always moved into `issue.md` so the ideas are still present, just in the correct place.
	- Mitigation: keep a regression test that asserts `spec.md` contains no injected validation text.
	- Rollback: revert the generator change.

## Rollout & Follow-up
- Release/rollout steps:
	- Merge the tooling change and run `Dev: 3 Create Active Folder` on a representative potential bug file to confirm correct output.
- Post-fix monitoring or clean-up tasks:
	- Rerun the generator tests when modifying templates under `docs/features/templates/bug/`.
- Links: issue, PRs, related docs
	- Issue: https://github.com/drmoisan/lexile-corpus-tuner/issues/102
