# codex-web-setup-crash (Issue #104)

- Date captured: 2026-01-21
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/codex-web-setup-crash/ (Issue #104)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #104
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/104
- Last Updated: 2026-01-21
## Summary

Codex web environment setup fails while installing system packages.
The bootstrap script exits with code 100 after an apt mirror returns a 502.

## Environment

- OS/version: Linux (Codex web job; Ubuntu noble mirror in apt output)
- Python version: Unknown
- Command/flags used: `.github/codex/codex-web-setup.sh`
- Data source or fixture: None

## Steps to Reproduce

1. Run a Codex web job that executes `.github/codex/codex-web-setup.sh`.
2. Wait for the system package install step.
3. Observe apt failing during package fetch.

## Expected Behavior

The setup script completes and the environment is fully provisioned.

## Actual Behavior

The script exits with code 100 during apt install.
Key error: `Failed to fetch ... 502 Bad Gateway` from the Ubuntu mirror.

## Logs / Screenshots

- [x] Attached minimal logs or screenshot
- Snippet:
	```
	E: Failed to fetch http://archive.ubuntu.com/ubuntu/pool/universe/n/node-cjs-module-lexer/node-cjs-module-lexer_1.2.3%2bdfsg-1_all.deb  502  Bad Gateway [IP: 172.31.1.19 8080]
	E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
	```

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

Likely transient 502 from the Ubuntu archive mirror or an upstream proxy.
Failure occurs during apt package downloads in `codex-web-setup.sh`.

## Proposed Fix / Validation Ideas

- [ ] Unit coverage areas
- [x] Integration scenario to retest
- [x] Manual verification notes

Integration scenario to retest: rerun the Codex web setup job after adding apt retries/timeout handling.
Manual verification notes: confirm the script completes and `shell-qc` tools install successfully.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch