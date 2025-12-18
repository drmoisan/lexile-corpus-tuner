# fix-all-script (Spec)

- Issue: #44
- Owner: Dan Moisan
- Date: 2025-12-15
- Status: Draft

## Context
Running `scripts/dev-tools/fix-all.ps1` aborts during the Black step with a "Black formatting failed" error even though Black reports files unchanged.

Environment:
- OS/version: Linux dev container (Debian bookworm)
- Python version: Not captured (using repo .venv from fix-all.ps1)
- Command/flags used: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File /workspaces/lexile-corpus-tuner/scripts/dev-tools/fix-all.ps1`
- Data source or fixture: None

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. From the repo root, run the fix-all task (`pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/fix-all.ps1`).
2. Observe the Black step output.
3. The script exits with a failure before continuing to later steps.

Expected:
fix-all.ps1 completes all steps (Black → Ruff → Pyright → Pytest) successfully and returns exit code 0.

Actual:
Black reports "All done! 98 files left unchanged." but fix-all.ps1 immediately emits `Black formatting failed. Please review errors above.` and exits with code 1, stopping the rest of the pipeline.

Logs / Screenshots:
- [x] Attached minimal logs or screenshot
- Snippet:

```
& /workspaces/lexile-corpus-tuner/.venv/bin/Activate.ps1
All done! ✨ 🍰 ✨
98 files left unchanged.
Write-Failure: /workspaces/lexile-corpus-tuner/scripts/dev-tools/fix-all.ps1:79:13
Black formatting failed. Please review errors above.
```


## Scope & Non-Goals
- In scope:
	- Investigate and fix `scripts/dev-tools/fix-all.ps1` Black step handling so a clean Black run does not fail the script.
	- Preserve existing Ruff, Pyright, and Pytest sequencing after Black.
	- Keep compatibility with existing VS Code tasks that invoke fix-all.ps1.
- Out of scope / non-goals:
	- Changing the order of the toolchain steps.
	- Adding new language/tool checks beyond the current pipeline.
	- Modifying Black/Ruff/Pyright configurations.

## Root Cause Analysis
- Observed behavior: Black reports success and unchanged files yet fix-all.ps1 reports "Black formatting failed" and exits non-zero.
- Hypothesis: fix-all.ps1 treats any stderr output or a mis-captured `$LASTEXITCODE` as failure; Black may emit to stderr even on success, or the script may overwrite the exit code after running Black (e.g., via activation or Write-* calls).
- Evidence: Log snippet shows Black success message followed immediately by Write-Failure from fix-all.ps1 around line 79 (Black block). No other errors printed.


## Proposed Fix
- Adjust Black invocation in fix-all.ps1 to correctly honor Black's exit code and ignore benign stderr when exit code is zero.
- Capture `$LASTEXITCODE` immediately after Black and avoid overwriting it before failure checks.
- Improve failure messaging to include Black exit code and stderr when the step truly fails.
- Keep existing behavior for non-zero exit codes to avoid masking real failures.
- Validation items:
	- [ ] Unit coverage areas (Pester) around Black step behaviors (success with stderr vs actual failure).
	- [ ] Integration scenario to retest end-to-end fix-all.
	- [x] Manual verification notes: Re-run fix-all.ps1 after adjustments and confirm Black → Ruff → Pyright → Pytest all execute and exit 0.


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access): Running inside repo dev container or equivalent environment with Poetry/venv available; Black installed via Poetry.
- Constraints (budget, performance, compatibility): Must remain compatible with current task wrappers and CI expectations; no material slowdown to fix-all sequence.
- External dependencies (services, libraries, releases): Depends on Black CLI behavior and exit codes; no external services.

## Data / API / Config Impact
- User-facing or API changes: None (developer tooling only).
- Data or migration considerations: None.
- Logging/telemetry updates: Potential improvement to failure messaging in fix-all.ps1.

## Test Strategy
- [ ] Unit coverage areas: Add Pester tests simulating Black exit code 0 with stderr, and non-zero with stderr, to ensure correct pass/fail handling.
- [ ] Integration scenario to retest: Run fix-all.ps1 end-to-end in the dev container.
- [x] Manual verification notes: Re-run fix-all.ps1 after adjustments and confirm it reaches Ruff, Pyright, Pytest, and exits 0.


## Acceptance Criteria
- Running fix-all.ps1 when Black reports success (even with unchanged files) completes the full sequence (Black → Ruff → Pyright → Pytest) and exits 0.
- If Black fails (non-zero exit), the script surfaces the exit code and stderr clearly and stops.
- Pester coverage guards against regressions where Black success is treated as failure due to stderr noise or exit-code mishandling.

## Risks & Mitigations
- Risks: Overly permissive handling could mask real Black issues; exit-code handling changes could affect later steps if shared variables are modified incorrectly.
- Mitigations: Limit logic changes to the Black block, keep failure on non-zero exit codes, add Pester coverage for success and failure cases.

## Rollout & Follow-up
- Release/rollout steps: Land fix on development branch, validate in dev container, ensure CI passes.
- Post-fix monitoring or clean-up tasks: Re-run fix-all.ps1 periodically; adjust messaging if developers report confusion.
- Links: issue #44; script scripts/dev-tools/fix-all.ps1; related VS Code tasks invoking fix-all.

