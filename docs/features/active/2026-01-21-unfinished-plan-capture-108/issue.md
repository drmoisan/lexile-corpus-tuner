# unfinished-plan-capture (Issue #108)

- Date captured: 2026-01-21
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/unfinished-plan-capture/ (Issue #108)

- Issue: #108
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/108
- Last Updated: 2026-01-22
## Problem / Why

We need a fast, deterministic way to answer: “Which active feature plans still have unchecked tasks?”

Right now, that information is scattered across many `docs/features/active/**/plan*.md` files, and it’s easy to miss incomplete tasks (especially in versioned subfolders like `v1/`, `v2/`). A single report artifact makes it easier to:

- spot in-progress work at a glance
- avoid “done-but-not-checked-off” plan drift
- prioritize review/cleanup for stale features

## Proposed Behavior

Add a Python dev-tool script that:

1. Recursively scans `docs/features/active` for plan files matching:
	- `plan.md`
	- `plan.*.md` (where `*` is a wildcard)
	- `plan-*.md` (dash variant used by some versioned folders)
2. For each plan file found, counts markdown task list checkboxes:
	- Checked: `- [x]` or `- [X]`
	- Unchecked: `- [ ]`
3. Excludes any plan where:
	- there are **no** checkboxes, or
	- **all** checkboxes are checked
4. Outputs a markdown artifact containing a table with columns:
	- feature
	- issue
	- type
	- remaining (unchecked/total)

Feature name resolution:

- If the plan file is directly under a folder that does **not** start with `v`, the feature name is that folder name.
- If the plan file is inside a folder whose name starts with `v` (e.g., `v1`, `v2`), the feature name is the **parent** folder name.

Type resolution:

- If the plan file is inside a `v*` folder, type is that `v*` folder name (e.g., `v2`).
- Otherwise, type is `base`.

Issue resolution:

- Prefer parsing an explicit header line like `- Issue: #77` or `- **Issue:** #77`.
- If missing, fall back to extracting a trailing `-<digits>` suffix from the feature folder name.

## Acceptance Criteria (early draft)

- [ ] Script exists as a Python module under `scripts/dev_tools/` and can be run to generate a report artifact (markdown).
- [ ] Script scans `docs/features/active` recursively for `plan.md` and `plan.*.md`.
- [ ] Script counts checkboxes and excludes complete plans (0 unchecked) and plans with no tasks.
- [ ] Output is a markdown table with columns: feature, issue, type, remaining (unchecked/total).
- [ ] Feature naming and type rules are implemented exactly:
	- [ ] if immediate parent folder starts with `v`, feature is the parent folder’s name
	- [ ] otherwise feature is the immediate parent folder name
- [ ] Unit tests exist for checkbox counting, issue parsing, feature/type resolution, and table rendering.

## Constraints & Risks

- Must be deterministic and fast (pure markdown parsing; no network).
- Keep logic testable without relying on filesystem temp files.
- Plan file formats vary slightly; issue parsing should tolerate both `Issue:` and `**Issue:**` header styles.
- Versioned plan naming is inconsistent (`plan.*.md` exists; some historical plans may use `plan-*.md`).

## Test Conditions to Consider

- [ ] Unit coverage areas
	- [ ] checkbox counting: checked vs unchecked; ignore non-list occurrences
	- [ ] issue parsing: `- Issue: #NN` and `- **Issue:** #NN`
	- [ ] feature/type resolution for versioned folders (`v1/`, `v2/`)
	- [ ] exclude complete plans and plans with zero checkboxes
	- [ ] markdown table rendering

- [ ] Integration scenarios
	- [ ] run against the real `docs/features/active` tree and confirm it produces an artifact

- [ ] CLI/API examples
	- [ ] `poetry run python -m scripts.dev_tools.plan_progress_report`
	- [ ] `poetry run python -m scripts.dev_tools.plan_progress_report --out artifacts/active_plan_progress.md`

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/unfinished-plan-capture/` folder from the template
