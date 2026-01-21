<!-- markdownlint-disable-file -->

# Task Research Notes: new-folder-squashes-spec-template (Issue #102)

## Research Executed

### File Analysis

- `docs/features/active/2026-01-21-new-folder-squashes-spec-template-102/issue.md`
  - Captures the bug report and expectations: the bug-folder generator should not pre-populate `spec.md` sections like `## Proposed Fix` and `## Test Strategy` from early hypotheses.
- `docs/features/templates/bug/spec.md`
  - Bug spec template defines a structured `## Proposed Fix` section with multiple `###` subsections, including `### Design summary (what changes where):`.
- `scripts/dev_tools/new_active_feature_folder.py`
  - Implements `Dev: 3 Create Active Folder` for `--type bug`; extracts `## Proposed Fix / Validation Ideas` from a potential bug file into `bug_validation` and contains logic that can seed parts of `spec.md`.
- `.vscode/tasks.json`
  - Contains the VS Code task label `Dev: 3 Create Active Folder`, which shells out to the Python module that creates active folders.

### Code Search Results

- `bug_validation`
  - 5 matches in `scripts/dev_tools/new_active_feature_folder.py` (e.g., the `bug_validation = sections.get("bug_validation", "").strip()` assignment and subsequent usage).
- `### Design summary (what changes where):`
  - 1 match in `docs/features/templates/bug/spec.md` (the template defines this subsection under `## Proposed Fix`).
- `Seeded from issue`
  - 1 match in `scripts/dev_tools/new_active_feature_folder.py` (the current codebase includes a prefix string used when seeding content into a section).
- `Dev: 3 Create Active Folder`
  - 1 match in `.vscode/tasks.json` (task label).

### External Research

- #fetch:https://github.com/drmoisan/lexile-corpus-tuner/issues/102
  - Confirmed the GitHub issue mirrors the local promoted `issue.md`, including the expectation that early hypotheses remain in `issue.md` and the final `spec.md` is filled after research.
- #fetch:https://github.com/drmoisan/lexile-corpus-tuner
  - Confirmed repository context and that the bug report is for the same repo.

### Project Conventions

- Standards referenced: Black (format), Ruff (lint), Pyright strict (type check), Pytest (tests)
- Instructions followed: `.github/instructions/general-code-change.instructions.md`, `.github/instructions/python-code-change.instructions.md`, `.github/instructions/general-unit-test.instructions.md`, `.github/instructions/python-unit-test.instructions.md`

## Key Discoveries

### Project Structure

Bug feature folders and templates live under:

- Templates: `docs/features/templates/bug/`
- Active bug folders: `docs/features/active/<slug>/`
- Generator: `scripts/dev_tools/new_active_feature_folder.py` (invoked by the VS Code task)

### Implementation Patterns

The generator’s behavior is driven by section extraction from a promoted potential bug markdown file:

- The potential bug file’s `## Proposed Fix / Validation Ideas` section is extracted via `get_section(...)` and stored as `sections["bug_validation"]`.
- In the bug flow, the generator uses `bug_validation` as *seed material* for parts of `spec.md`.

The bug spec template itself is intentionally structured and “prompt-like”, with a dedicated subsection:

- `### Design summary (what changes where):`

This subsection is intended to be filled later (often after additional research), not populated automatically from early hypotheses.

### Complete Examples

```python
# Evidence (generator extracts issue/potential content into a dedicated key).
# From: scripts/dev_tools/new_active_feature_folder.py
sections: dict[str, str] = {
    # ...
    "bug_validation": get_section(potential_content, "Proposed Fix / Validation Ideas"),
}
```

```markdown
<!-- Evidence (template subsection exists and is intended to be filled later). -->
<!-- From: docs/features/templates/bug/spec.md -->

## Proposed Fix

### Design summary (what changes where):
```

### API and Schema Documentation

No external API/schema changes are required for this issue. This is a documentation/template integrity bug in the active-folder generator.

### Configuration Examples

```text
VS Code task label: Dev: 3 Create Active Folder
Underlying command pattern:
  poetry run python -m scripts.dev_tools.new_active_feature_folder \
    --feature-name <bug-name> --type bug --issue-number <issue|auto>
```

### Technical Requirements

- For `--type bug`:
  - Preserve the bug `spec.md` template structure exactly (including all `###` subsections).
  - Do not insert early hypothesis content from `issue.md` into `## Proposed Fix`.
  - Specifically: do not auto-fill `### Design summary (what changes where):` from `issue.md`.
  - (Per issue #102) prefer that early hypotheses remain only in `issue.md` until research is complete.
- Unit tests must not create temp files (repo policy). Use an in-memory filesystem approach.
- Must pass full Python toolchain (Black → Ruff → Pyright → Pytest).

**Mandatory unachievable objective callout**:
- None identified.

## Recommended Approach

Treat `issue.md` as the “early thinking + evidence capture” document and keep `spec.md` as the “post-research design commitment” document.

Concretely for the bug-folder generator:

1. Continue seeding *factual* sections into `spec.md` (these are not design-commitment sections):
   - `## Context` (summary / environment / impact)
   - `## Repro & Evidence` (repro steps, expected vs actual, logs)
   - `## Root Cause Analysis` (suspected cause notes)

2. Do not seed any content derived from `## Proposed Fix / Validation Ideas` into `spec.md` under:
   - `## Proposed Fix`
   - `## Test Strategy`

3. Ensure the promoted potential file is still moved into the active folder as `issue.md` so the hypotheses are preserved and the LLM can later synthesize them into the spec.

This recommendation directly matches the intent stated in Issue #102 and avoids prematurely “locking in” a design direction.

## Implementation Guidance

- **Objectives**:
  - Preserve the bug spec template’s prompts and subsection structure.
  - Ensure `### Design summary (what changes where):` remains unpopulated by automation.
  - Keep early hypotheses in `issue.md`, not in `spec.md`.

- **Key Tasks**:
  - Update `scripts/dev_tools/new_active_feature_folder.py` bug flow to skip applying `bug_validation` to `spec.md` (Proposed Fix + Test Strategy).
  - Add/adjust unit tests under `tests/scripts/dev_tools/` to assert:
    - Template headings/subsections remain.
    - `bug_validation` text is *not present* in `## Proposed Fix` and `## Test Strategy`.
    - The resulting `issue.md` still contains the early hypothesis content.
  - Manual validation: run `Dev: 3 Create Active Folder` against a potential bug file containing non-trivial `## Proposed Fix / Validation Ideas` and verify the resulting `spec.md` is still “blank prompts only” in the design sections.

- **Dependencies**:
  - None (pure Python + existing repo tooling).

- **Success Criteria**:
  - Generator produces `docs/features/active/<slug>/spec.md` with unchanged template structure and no injected `bug_validation` content in `## Proposed Fix` / `## Test Strategy`.
  - `docs/features/active/<slug>/issue.md` contains the early hypotheses verbatim.
  - Toolchain pass: Black, Ruff, Pyright, Pytest all succeed.
