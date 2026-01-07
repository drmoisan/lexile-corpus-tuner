# Bugfix Playbook

A disciplined workflow for capturing, triaging, fixing, verifying, and archiving bugs in this repo.

Key paths:
- Backlog (ideas/features): `docs/features/backlog.md`
- Active features (if the bug is tied to one): `docs/features/active/<feature>/`
- Tests: `tests/` (unit/integration), optionally `tests/bugs/<YYYY>/<issue>-<desc>.py` for standalone bug regressions

Key scripts/tasks:
- Use `gh issue create` (or GitHub UI) for the bug record.
- VS Code tasks: Run checks (`Run All Checks`, `Fix All`) and feature helpers if the bug touches an active feature (e.g., `link-feature-docs.ps1` to add links).
- Local quality gates: JSON formatting/validation, shell QC (shfmt/shellcheck/bats), Black, Ruff, Pyright, Pytest, and PoshQC (`poetry run python -m scripts.dev_tools.fix_all` runs them in sequence; PowerShell wrapper available at `scripts/dev-tools/fix-all.ps1`).

---

## 1) Capture & issue
- Create a GitHub issue for any non-trivial bug before coding.
- Include: Problem summary, expected vs actual, repro steps (inputs/CLI/env), evidence (stack trace/logs/screenshots/output), severity, scope, and test conditions to add.
- If the bug belongs to an active feature, note the feature folder and issue link.

## 2) Triage
- Decide severity/priority and ownership.
- Record the triage outcome in the issue (don’t keep private notes).

## 3) Branch
- Use `bugfix/<short>-#<issue>` (e.g., `bugfix/speakerless-over-split-#31`).

## 4) Write a failing test first
- Add a regression test that reproduces the bug (minimal, deterministic).
- Placement: `tests/bugs/<YYYY>/<issue>-<desc>.py` or the relevant area (`tests/transform/`, `tests/integration/`, etc.).
- The test should fail before the fix and pass after.

## 5) Implement the fix
- Keep the change minimal and targeted.
- Respect existing ETL boundaries; avoid speculative refactors.
- Keep Black/Ruff/Pyright clean; log only where useful.
- If you uncover deeper design issues, open a new issue rather than expanding scope.

## 6) Local verification
- Run `poetry run python -m scripts.dev_tools.fix_all` (or the `Run All Checks` VS Code task; PowerShell wrapper at `scripts/dev-tools/fix-all.ps1`) for JSON/shell QC → Black/Ruff/Pyright/Pytest → PoshQC.
- Run any targeted integration/end-to-end checks relevant to the bug.
- Validate against the original repro (and sample transcripts if applicable).

## 7) Pull request
- Title: `fix: correct <thing> in <module> (fixes #<issue>)`
- Body: summary of the bug/fix, root cause analysis, link to regression test, risk/impact, behavior/migration notes if any.

## 8) Review
- Ensure the regression test captures the prior failure.
- Validate behavior against the issue description; push back on missing tests.
- If deeper problems surface, open a new issue; don’t expand PR scope.

## 9) Merge & follow-up
- Merge to `development`; ensure “fixes #<issue>` auto-closes.
- Update any linked feature docs if the bug was tied to an active feature.
- Leave the regression test in place; update backlog/status if needed.

## Anti-patterns
- Fixing without an issue.
- Shipping a fix without a failing-then-passing test.
- Mixing unrelated fixes or refactors.
- Skipping Pyright/Ruff/Black/tests.
- Keeping discussion outside the issue/PR.
