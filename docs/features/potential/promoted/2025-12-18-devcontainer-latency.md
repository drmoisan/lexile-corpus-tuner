# devcontainer-latency (Issue #55)

- Date captured: 2025-12-18
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/devcontainer-latency/ (Issue #55)

- Issue: #55
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/55
- Last Updated: 2025-12-19
## Summary

Pytest collection in the VS Code dev container is ~8× slower (11–12s) than on the host (~1.53s) because the Task Explorer extension causes heavy workspace scanning and filesystem I/O contention.

## Environment

- OS/version: VS Code dev container on Linux host (workspace mounted from host)
- Python version: 3.12.x (Poetry environment)
- Command/flags used: `poetry run pytest --collect-only` (also standard pytest runs)
- Data source or fixture: Lexile Corpus Tuner repo workspace

## Steps to Reproduce

1. Open the repo inside the VS Code dev container with the Task Explorer extension enabled.
2. Run `poetry run pytest --collect-only` (or a normal pytest run) from the workspace root.
3. Observe collection taking ~11–12 seconds.
4. Disable the Task Explorer extension and rerun to confirm collection time drops substantially.

## Expected Behavior

Pytest collection and startup in the dev container should be close to host performance (around 1–2 seconds for ~521 tests) without significant extension-induced overhead.

## Actual Behavior

Pytest collection inside the dev container consistently takes ~11–12 seconds for ~521 tests, compared to ~1.53 seconds on the host. Disabling Task Explorer restores normal performance.

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet:
	- Dev container: `poetry run pytest --collect-only` → ~11–12s
	- Host: `poetry run pytest --collect-only` → ~1.53s
	- With `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, restricted discovery, cache_dir=/tmp, and --noconftest: still ~11–12s

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

Root cause: Task Explorer extension continuously scans the workspace to discover tasks; on the dev container’s mounted filesystem this creates heavy I/O contention, making pytest’s metadata-heavy collection far slower.

Contributing factors:
- Workspace mounted from host into the dev container (higher FS latency).
- Task Explorer background scans overlapping with pytest file metadata reads.
- Large repository with many Python files.

Timeline (abridged):
- Baseline: ~11–12s collection in container.
- Disabled pytest autoload/plugins/scopes/cache/conftest: no improvement (still ~11s; --noconftest slower).
- Host comparison: ~1.53s.
- Disabling Task Explorer: restores normal performance → confirmed source of contention.

## Proposed Fix / Validation Ideas

Actions/resolution:
- Migrate the *code* to WSL2 (small) 
- Keep *artifacts* on externally blind mounted directory (large)

Validation:
- Re-run `poetry run pytest --collect-only` to confirm collection time returns to ~1–2s.
- Spot-check a full pytest run to ensure end-to-end performance is improved.

## Next Step

- [x] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch