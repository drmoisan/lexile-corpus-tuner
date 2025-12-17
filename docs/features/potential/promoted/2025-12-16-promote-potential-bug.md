# promote-potential-bug (Issue #50)

- Date captured: 2025-12-16
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/promote-potential-bug_Potential_Bug/ (Issue #50)

- Issue: #50
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/50
- Last Updated: 2025-12-17
## Summary

Promoting a potential bug file to GitHub creates an issue labeled `bug` but the issue body is populated with feature-template placeholders rather than the details from the source markdown.

## Environment

- OS/version: Linux dev container (Debian bookworm)
- Python version: repo .venv used by promotion task (version not printed)
- Command/flags used: VS Code task `Dev: 2 Promote Potential Issue to GitHub Issue` (poetry run python -m scripts.dev_tools.potential_to_issue --potential-path <file> --promotion-type bug)
- Data source or fixture: docs/features/potential/promoted/2025-12-15-fix-all-script.md

## Steps to Reproduce

1. Run the promotion task against docs/features/potential/2025-12-15-fix-all-script.md with `--promotion-type bug`.
2. Open GitHub issue #44 created by the tooling.
3. Compare the issue body to the source markdown content.

## Expected Behavior

The created issue body should mirror the bug template fields (summary, environment, steps, expected/actual behavior, impact) from the potential file, and the issue title should reflect the bug (e.g., `Bug: fix-all-script`).

## Actual Behavior

Issue #44 is labeled `bug`, but its body consists entirely of feature-template placeholder sections (Problem / Why, Proposed Behavior, etc.) with `(not provided in potential file)` values, and the title is `Feature: fix-all-script` instead of a bug-specific title.

## Logs / Screenshots

- [x] Attached minimal logs or screenshot
- Snippet: Issue #44 body shows only placeholder sections while docs/features/potential/promoted/2025-12-15-fix-all-script.md contains full bug details (summary, environment, repro steps, expected/actual).

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

- The promotion script always prefixes the title with `Feature:` regardless of promotion_type, so bug issues inherit a feature-style title ([scripts/dev_tools/potential_to_issue.py#L304](scripts/dev_tools/potential_to_issue.py#L304)).
- The issue body is assembled from feature headings (Problem / Why, Proposed Behavior, Acceptance Criteria, Constraints & Risks, Test Conditions), but the bug template uses different headings (Summary, Environment, Steps to Reproduce, Expected/Actual), causing every section to fall back to the `(not provided in potential file)` placeholder ([scripts/dev_tools/potential_to_issue.py#L195-L196](scripts/dev_tools/potential_to_issue.py#L195-L196), [scripts/dev_tools/potential_to_issue.py#L306-L310](scripts/dev_tools/potential_to_issue.py#L306-L310)).
- Source bug file docs/features/potential/promoted/2025-12-15-fix-all-script.md contains the correct data, so the data loss happens during extraction, not authoring.

## Proposed Fix / Validation Ideas

Phase 1 – Parsing & Title Logic
- Detect promotion type `bug` and map bug-template headings (Summary, Environment, Steps to Reproduce, Expected Behavior, Actual Behavior, Impact / Severity, Logs) into the issue body. Preserve feature/refactor flows for other promotion types.
- Adjust issue_title to align with promotion type (e.g., `Bug: <name>` when promotion_type == "bug").
- Add unit tests that cover bug and feature templates to ensure extraction populates the body without placeholders.

Phase 2 – CLI/Task Wiring
- Update the promotion task (Dev: 2 Promote Potential Issue to GitHub Issue) to pass the correct promotion type through and, if needed, an explicit template selector.
- Document the accepted headings for each promotion type to keep templates and parser in sync.

Phase 3 – Remediate Existing Issue #44
- After fixes land, re-run the promotion on docs/features/potential/promoted/2025-12-15-fix-all-script.md to regenerate the correct issue body, or manually update issue #44 using the extracted fields to match the source markdown.
- Verify the updated issue body matches the source content and retains the `bug` label.

Phase 4 – Promote This Bug with Fixed Tooling
- Once the parser and task are fixed, promote this file to a GitHub issue using the corrected promotion flow and confirm the body matches this analysis.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch