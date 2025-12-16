# fix-all-script (Potential Bug) (Issue #44)

- Date captured: 2025-12-15
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/fix-all-script_Potential_Bug/ (Issue #44)

- Issue: #44
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/44
- Last Updated: 2025-12-16
## Summary

Running `scripts/dev-tools/fix-all.ps1` aborts during the Black step with a "Black formatting failed" error even though Black reports files unchanged.

## Environment

- OS/version: Linux dev container (Debian bookworm)
- Python version: Not captured (using repo .venv from fix-all.ps1)
- Command/flags used: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File /workspaces/lexile-corpus-tuner/scripts/dev-tools/fix-all.ps1`
- Data source or fixture: None

## Steps to Reproduce

1. From the repo root, run the fix-all task (`pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/fix-all.ps1`).
2. Observe the Black step output.
3. The script exits with a failure before continuing to later steps.

## Expected Behavior

fix-all.ps1 completes all steps (Black → Ruff → Pyright → Pytest) successfully and returns exit code 0.

## Actual Behavior

Black reports "All done! 98 files left unchanged." but fix-all.ps1 immediately emits `Black formatting failed. Please review errors above.` and exits with code 1, stopping the rest of the pipeline.

## Logs / Screenshots

- [x] Attached minimal logs or screenshot
- Snippet:

```
& /workspaces/lexile-corpus-tuner/.venv/bin/Activate.ps1
All done! ✨ 🍰 ✨
98 files left unchanged.
Write-Failure: /workspaces/lexile-corpus-tuner/scripts/dev-tools/fix-all.ps1:79:13
Black formatting failed. Please review errors above.
```

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

Black appears to exit cleanly but the script treats the step as a failure; investigate fix-all.ps1 Black invocation/exit code handling and any redirected error output around line 79.

## Proposed Fix / Validation Ideas

- [ ] Unit coverage areas
- [ ] Integration scenario to retest
- [x] Manual verification notes
	- Re-run fix-all.ps1 after adjusting Black invocation/error handling and confirm all steps proceed.

## Next Step

- [x] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch