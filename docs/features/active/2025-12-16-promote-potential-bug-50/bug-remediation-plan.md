# Bug Promotion Remediation Plan

## **Overview**
Create a corrected promotion flow so potential bug markdown files generate GitHub issues with the proper bug template content and titles, fix the existing mislabeled Issue #44, and then promote the current bug report using the repaired tooling.

## **Phase 0 — Context & Inputs**

- [ ] [P0-T1] Re-read scripts/dev_tools/potential_to_issue.py to catalog current parsing logic, title construction, and promotion_type handling for feature vs bug flows.
- [ ] [P0-T2] Review docs/features/potential/promoted/2025-12-15-fix-all-script.md and GitHub Issue #44 side-by-side to confirm the exact data loss cases.
- [ ] [P0-T3] Identify existing tests (if any) covering promotion tooling to understand current coverage and patterns to extend.

## **Phase 1 — Design Adjustments**

- [ ] [P1-T1] Define field mapping for promotion_type == "bug" (Summary, Environment, Steps to Reproduce, Expected Behavior, Actual Behavior, Logs/Screenshots, Impact/Severity) and decide how to render these into the issue body.
- [ ] [P1-T2] Decide title schema per promotion_type (e.g., "Bug: <name>", "Feature: <name>", "Refactor: <name>", "Epic: <name>") and document fallback behavior.
- [ ] [P1-T3] Specify how to handle missing sections gracefully (retain placeholders vs omit empty sections) and record the decision for implementation.

## **Phase 2 — Implementation Changes**

- [ ] [P2-T1] Update promote_potential in scripts/dev_tools/potential_to_issue.py to generate titles based on promotion_type rather than always using "Feature:".
- [ ] [P2-T2] Implement bug-template extraction in promote_potential for promotion_type == "bug" using the bug headings defined in [P1-T1].
- [ ] [P2-T3] Preserve existing feature/refactor/epic behavior while ensuring the new bug path does not regress other promotion types.
- [ ] [P2-T4] Add or update CLI/task wiring (Dev: 2 Promote Potential Issue to GitHub Issue) so the promotion_type argument is correctly passed and documented for bug promotions.

## **Phase 3 — Automated Tests (scenario-specific)**

- [ ] [P3-T1] Add unit test in tests/dev_tools/test_potential_to_issue.py (or new file) that verifies promote_potential builds a bug issue body using Summary/Environment/Steps/Expected/Actual/Impact headings from a sample bug markdown, with no `(not provided...)` placeholders when data exists.
- [ ] [P3-T2] Add unit test covering bug title generation to confirm "Bug: <name>" is used when promotion_type == "bug".
- [ ] [P3-T3] Add regression test ensuring feature promotion still uses the feature headings and preserves existing body structure and title prefix.
- [ ] [P3-T4] Add unit test that validates missing bug sections fall back to the chosen behavior from [P1-T3] (e.g., placeholders or omission) to lock in future expectations.

## **Phase 4 — Verification & Toolchain**

- [ ] [P4-T1] Run Black, Ruff, Pyright, and Pytest to validate formatting, linting, typing, and tests after code changes.
- [ ] [P4-T2] Document any notable behavior changes or new headings in README or relevant docs for promotion tooling usage.

## **Phase 5 — Remediate Issue #44**

- [ ] [P5-T1] Re-run the fixed promotion workflow against docs/features/potential/promoted/2025-12-15-fix-all-script.md to regenerate the issue body, or manually update Issue #44 using the corrected content extracted from the source file.
- [ ] [P5-T2] Confirm Issue #44 retains the `bug` label and now matches the source bug markdown fields (summary, environment, steps, expected/actual, logs, impact).

## **Phase 6 — Promote Current Bug with Fixed Tooling**

- [ ] [P6-T1] Using the repaired promotion flow, promote docs/features/potential/2025-12-16-promote-potential-bug.md as a bug and verify the created issue body matches the file’s content and uses the correct title prefix.
- [ ] [P6-T2] Move the promoted file to the appropriate promoted folder and update metadata as per the tooling’s normal flow.