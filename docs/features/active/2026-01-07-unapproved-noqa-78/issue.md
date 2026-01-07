# unapproved-noqa (Issue #78)

- Date captured: 2026-01-07
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/unapproved-noqa/ (Issue #78)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #78
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/78
- Last Updated: 2026-01-07
## Summary

Noqa suppressions are ad hoc: multiple recurring patterns lack deterministic, pre-authorized rules, so suppressions are being added inconsistently without a policy.

## Environment

- OS/version: Windows 11 Pro 10.0.26200
- Python version: 3.13.7
- Command/flags used: internal one-time audit by an agent (no rerunnable tooling yet)
- Data source or fixture: agent audit findings (one-time)

## Steps to Reproduce

1. Review current `# noqa` suppressions across the codebase.
2. Note recurring patterns that are not covered by any deterministic, pre-authorized policy.
3. Observe inconsistent or ad hoc suppressions where guidance is missing.

## Expected Behavior

All recurring suppression patterns are documented with deterministic, pre-authorized rules (including required contexts and comment formats), so suppressions are consistent and policy-compliant.

## Actual Behavior

Current suppressions include recurring patterns that lack policy coverage, leading to ad hoc usage. Specific gaps include:
- Missing pre-authorized patterns for: ARG002 (unused args in test mocks), B008 (Typer defaults), TCH002/TCH003 (runtime + typing imports), S310 (urllib to trusted HTTPS endpoints), S314 (trusted XML parsing), BLE001 (CLI top-level catch-all), S301 (trusted pickle loads), S108/S105 (test literals).
- Non-authorized patterns needing code changes instead of suppression: TID252 (parent-relative imports), S607 (partial executable path), D401 (non-imperative docstrings), F401 (unused import), UP017 (naive datetime), plus ANN401 cases that need design review.

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet: One-time agent audit identified the patterns listed under “Actual Behavior”; no reusable tooling is available to re-run.

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

Suppression policy wasn’t updated after new recurring patterns emerged; current instructions lack deterministic, pre-authorized entries and required comment formats for those patterns.

## Proposed Fix / Validation Ideas

- [ ] Unit coverage areas
- [ ] Integration scenario to retest
- [x] Manual verification notes
  - Author deterministic suppression policy covering the recurring and justifiable patterns with required contexts and comment formats.
  - Define “not authorized—fix code instead” guidance for TID252, S607, D401, F401, UP017 and outline ANN401 evaluation criteria.
  - Create a repeatable audit or checklist to validate suppressions against the new policy and track required refactors.

## Next Step

- [x] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch