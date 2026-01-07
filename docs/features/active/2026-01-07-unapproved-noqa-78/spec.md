# 2026-01-07-unapproved-noqa (Spec)

- Issue: #78
- Owner: 2026-01-07-unapproved-noqa
- Date: 2026-01-07
- Status: Complete

## Context
Noqa suppressions are ad hoc: multiple recurring patterns lack deterministic, pre-authorized rules, so suppressions are being added inconsistently without a policy.

Environment:
- OS/version: Windows 11 Pro 10.0.26200
- Python version: 3.13.7
- Command/flags used: internal one-time audit by an agent (no rerunnable tooling yet)
- Data source or fixture: agent audit findings (one-time)

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Review current `# noqa` suppressions across the codebase.
2. Note recurring patterns that are not covered by any deterministic, pre-authorized policy.
3. Observe inconsistent or ad hoc suppressions where guidance is missing.

Expected:
All recurring suppression patterns are documented with deterministic, pre-authorized rules (including required contexts and comment formats), so suppressions are consistent and policy-compliant.

Actual:
Current suppressions include recurring patterns that lack policy coverage, leading to ad hoc usage. Specific gaps include:
- Missing pre-authorized patterns for: ARG002 (unused args in test mocks), B008 (Typer defaults), TCH002/TCH003 (runtime + typing imports), S310 (urllib to trusted HTTPS endpoints), S314 (trusted XML parsing), BLE001 (CLI top-level catch-all), S301 (trusted pickle loads), S108/S105 (test literals).
- Non-authorized patterns needing code changes instead of suppression: TID252 (parent-relative imports), S607 (partial executable path), D401 (non-imperative docstrings), F401 (unused import), UP017 (naive datetime), plus ANN401 cases that need design review.

Logs / Screenshots:
- [ ] Attached minimal logs or screenshot
- Snippet: One-time agent audit identified the patterns listed under “Actual Behavior”; no reusable tooling is available to re-run.


## Scope & Non-Goals
- In scope:
  - Document 11 pre-authorized suppression patterns with deterministic rules, required contexts, and comment formats
  - Document 5 non-authorized patterns with recommended code-change workarounds
  - Fix all unjustified suppressions using documented workarounds (TID252, S607, D401, F401, UP017)
  - Update test mocks to match new implementation patterns (shutil.which() validation, platform detection)
  - Regenerate AGENTS.md with complete suppression governance
  - Validate fixes through full toolchain (Black, Ruff, Pyright, Pytest, PowerShell)
- Out of scope / non-goals:
  - ANN401 (Any type) evaluation deferred - requires case-by-case design review
  - Comment format audit for existing pre-authorized patterns (can be done as follow-up)
  - Automated tooling to detect policy violations (manual audit sufficient for now)

## Root Cause Analysis
Suppression policy wasn’t updated after new recurring patterns emerged; current instructions lack deterministic, pre-authorized entries and required comment formats for those patterns.


## Proposed Fix
- [ ] Unit coverage areas
- [ ] Integration scenario to retest
- [x] Manual verification notes
  - Author deterministic suppression policy covering the recurring and justifiable patterns with required contexts and comment formats.
  - Define “not authorized—fix code instead” guidance for TID252, S607, D401, F401, UP017 and outline ANN401 evaluation criteria.
  - Create a repeatable audit or checklist to validate suppressions against the new policy and track required refactors.


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
  - Python 3.13.7 with strict Pyright type checking enabled
  - Ruff linter enforcing all rules without file-level exclusions
  - Test suite must remain at ≥90% coverage after fixes
- Constraints (budget, performance, compatibility):
  - Line length limit of 88 characters (Black default) - some absolute imports require E501 suppression
  - Must maintain backward compatibility with existing test fixtures and mocks
  - Platform-specific behavior (clipboard detection) requires conditional logic
- External dependencies (services, libraries, releases):
  - pyperclip (optional) lacks py.typed marker, requires import-untyped suppression
  - shutil.which() used for cross-platform executable validation
  - Typer framework pattern requires B008 suppression for Option() in signatures

## Data / API / Config Impact
- User-facing or API changes:
  - None - all changes are internal code quality improvements
  - Clipboard detection now uses explicit platform detection (more predictable behavior)
- Data or migration considerations:
  - None - no data format or schema changes
- Logging/telemetry updates (if any):
  - Enhanced error messages for git executable not found (FileNotFoundError with clear message)
  - Better docstrings provide improved logging context for debugging

## Test Strategy
- [x] Unit coverage areas
  - Updated 11 test files to work with new implementation patterns
  - Added proper type hints to test mocks (replaced lambdas with typed functions)
  - Mocked shutil.which() in all subprocess-related tests
  - Mocked sys.platform for platform-specific clipboard tests
  - All 1156 tests passing with 92% coverage (improved from 87%)
- [x] Integration scenario to retest
  - Full toolchain validation: JSON → Shell → Black → Ruff → Pyright → Pytest → PowerShell
  - All branches pass without errors or warnings
  - Coverage maintained above 90% threshold
- [x] Manual verification notes
  - Created comprehensive suppression policy in .github/instructions/python-suppressions.instructions.md
  - Documented 11 pre-authorized patterns with deterministic rules and required comment formats
  - Documented 5 non-authorized patterns with recommended workarounds
  - Applied all workarounds: converted relative imports to absolute, added shutil.which() validation, fixed docstring mood, removed unnecessary suppressions
  - Regenerated AGENTS.md with complete policy
  - Created artifacts/noqa-audit-report.md with full pattern analysis


## Acceptance Criteria
- [x] All recurring suppression patterns are documented in .github/instructions/python-suppressions.instructions.md with:
  - Deterministic authorization rules (when pre-authorized vs. requires user approval)
  - Required context and justification for each pattern
  - Required comment format (e.g., "# noqa: S603 - static analysis can't verify runtime validation")
  - Clear distinction between pre-authorized patterns and non-authorized patterns requiring code changes
- [x] All unjustified suppressions are fixed using documented workarounds:
  - TID252: 3 relative imports converted to absolute imports (with E501 for line length)
  - S607: Added shutil.which() validation to collect_commit_context.py
  - D401: Fixed 3 docstrings to use imperative mood
  - F401: Removed unnecessary suppression from pr_context/collector.py
  - UP017: Removed unnecessary suppression from summary_helpers.py (code already correct)
- [x] All tests pass with maintained or improved coverage:
  - 1156 tests passing (0 failures)
  - 92% coverage (improved from 87%)
  - Full toolchain passes: Black, Ruff, Pyright, Pytest, PowerShell
- [x] Documentation is complete and accurate:
  - AGENTS.md regenerated with all policy updates
  - artifacts/noqa-audit-report.md documents full pattern analysis
  - Pre-authorized patterns: S603, ARG002, B008, TCH002/TCH003, S310, S314, BLE001, S301, S108/S105, import-untyped, E501
  - Non-authorized with workarounds: S110, TID252, S607, D401, F401, UP017
- [x] Test mocks updated to match new implementation patterns:
  - 11 test files updated for shutil.which() validation
  - Platform-specific tests mock sys.platform for deterministic behavior
  - Type hints added to all test mocks (no lambda functions with unknown types)

## Risks & Mitigations
- Technical or operational risks:
  - Risk: Absolute imports exceeding 88-char line length could proliferate
    - Mitigation: E501 suppression allowed only for legitimately unavoidable cases; document that PEP 8 permits this
  - Risk: shutil.which() validation could break in environments where git is not on PATH
    - Mitigation: Clear FileNotFoundError with actionable message; documents requirement
  - Risk: Platform detection logic could fail in unexpected environments (e.g., new WSL versions)
    - Mitigation: Explicit platform checks with fallback chain; covered by tests
  - Risk: Pyright strict mode could flag legitimate use of dynamic imports or Any types
    - Mitigation: ANN401 evaluation deferred for case-by-case review; only justify when truly unavoidable
- Mitigations and rollbacks:
  - All changes are backwards compatible - no breaking API changes
  - Full test coverage ensures no regressions
  - If issues arise, suppressions can be temporarily re-added with user approval while proper fix is developed
  - Git history preserves all context for rollback if needed

## Rollout & Follow-up
- Release/rollout steps:
  - [x] Policy documentation complete (.github/instructions/python-suppressions.instructions.md)
  - [x] All code fixes applied and validated
  - [x] Full toolchain passes (Black, Ruff, Pyright, Pytest, PowerShell)
  - [x] AGENTS.md regenerated with complete policy
  - [ ] Create PR with all changes
  - [ ] Close issue #78 after PR merge
- Post-fix monitoring or clean-up tasks:
  - [ ] Optional: Audit ~60 existing pre-authorized suppressions to verify comment format consistency
  - [ ] Optional: Evaluate 3 ANN401 (Any type) instances for potential design improvements
  - [ ] Optional: Consider automated tooling to enforce suppression policy in CI
  - [ ] Optional: Document S110 pattern deprecation in migration guide for other contributors
- Links: issue, PRs, related docs
  - Issue: #78 (https://github.com/drmoisan/lexile-corpus-tuner/issues/78)
  - Branch: feature/populate-open-stax-ck-12-manifest-#73
  - Policy: .github/instructions/python-suppressions.instructions.md
  - Audit: artifacts/noqa-audit-report.md
  - Generated: AGENTS.md
