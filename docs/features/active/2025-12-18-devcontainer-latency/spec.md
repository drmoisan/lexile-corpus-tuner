# 2025-12-18-devcontainer-latency (Spec)

- Issue: #55
- Owner: 2025-12-18-devcontainer-latency
- Date: 2025-12-22
- Status: Draft

## Context
Pytest collection in the VS Code dev container was ~8× slower (11–12s) than on the host (~1.53s) due to aggressive workspace scanning causing filesystem I/O contention on the mounted filesystem inside the dev container. Pytest’s metadata-heavy collection amplified this cost. The problematic scanner has already been disabled; the goal now is to prevent regressions and harden the container configuration.

Environment:
- VS Code dev container on Linux host (workspace bind-mounted from host)
- Python 3.12.x (Poetry)
- Commands: `poetry run pytest --collect-only` (and standard pytest runs)
- Workspace: Lexile Corpus Tuner repo

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce (historical):
1. Open the repo inside the VS Code dev container with an aggressive workspace-scanning extension enabled.
2. Run `poetry run pytest --collect-only` (or a normal pytest run) from the workspace root.
3. Observe collection taking ~11–12 seconds.
4. Disable the scanning extension and rerun to confirm collection time drops substantially.

Expected:
Pytest collection and startup in the dev container should be close to host performance (around 1–2 seconds for ~521 tests) without significant extension-induced overhead.

Actual (before mitigation):
Pytest collection inside the dev container consistently took ~11–12 seconds for ~521 tests, compared to ~1.53 seconds on the host. Disabling the scanning extension restored normal performance.

Logs / Screenshots:
- [ ] Attached minimal logs or screenshot
- Snippet:
	- Dev container: `poetry run pytest --collect-only` → ~11–12s
	- Host: `poetry run pytest --collect-only` → ~1.53s
	- With `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, restricted discovery, cache_dir=/tmp, and --noconftest: still ~11–12s (no improvement)


## Scope & Non-Goals
- In scope: VS Code devcontainer performance fixes; extension configuration; filesystem/mount layout guidance; validation benchmarks; documentation for container users.
- Out of scope / non-goals: Pytest optimizations unrelated to filesystem I/O; broader CI/CD performance; non-devcontainer host tuning.

## Root Cause Analysis
Root cause: An aggressive workspace-scanning extension continuously scanned the workspace; in a bind-mounted devcontainer this created heavy filesystem I/O contention, slowing pytest collection.

Contributing factors:
- Devcontainer workspace mounted from host (higher FS latency).
- Background scanning overlapping with pytest’s many stat/reads.
- Large Python codebase amplifying metadata reads.

Timeline (abridged):
- Baseline: ~11–12s collection in container.
- Disabled pytest autoload/plugins/scopes/cache/conftest: no improvement (still ~11s; --noconftest slower).
- Host comparison: ~1.53s.
- Disabling the scanning extension: performance improves → confirmed cause.


## Proposed Fix
- Keep aggressive workspace-scanning extensions disabled in the devcontainer extension set; document recommended/allowed extensions.
- Keep source code on a low-latency path (WSL2 filesystem or container volume). Bind-mount large artifacts separately.
- Update `devcontainer.json` (or docs) with mounts guidance and an extension recommendation/exclusion list.
- Add usage guidance: avoid dual writers (host + container), and use save-all before agent runs.

Validation:
- Re-run `poetry run pytest --collect-only` targeting ~1–2s for ~521 tests.
- Run a representative short pytest suite to confirm end-to-end improvement.
- Record before/after timings and environment in docs.


## Assumptions, Constraints, Dependencies
- Assumptions: Access to adjust devcontainer settings; ability to control extension set; host supports WSL2/volume storage.
- Constraints: Must retain host-mounted artifacts; no new runtime deps; keep workflows consistent with existing devcontainer tooling.
- Dependencies: VS Code devcontainers; Docker/WSL2; pytest/Poetry environment.

## Data / API / Config Impact
- User-facing/API changes: None.
- Data/migration: None.
- Config: Devcontainer settings and extension recommendations; optional mounts guidance.

## Test Strategy
- Baseline: capture current `poetry run pytest --collect-only` and a short targeted test run (record timings/environment).
- After changes: re-run the same commands; compare timings aiming for ~1–2s collection.
- Verify extension set in container matches expectations (aggressive scanners remain disabled).
- Document measurements in the feature docs.


## Acceptance Criteria
- Pytest `--collect-only` in devcontainer completes ~1–2s for ~521 tests (or materially matches host within small delta).
- Short pytest run shows improved runtime vs baseline.
- Aggressive workspace-scanning extensions remain disabled by default in the container environment or are excluded via recommendations.
- Docs updated with mounts/layout guidance and troubleshooting steps.

## Risks & Mitigations
- Risk: Users re-enable aggressive scanning extensions. Mitigation: document exclusions; provide recommended extension set.
- Risk: Mount changes disrupt artifact access. Mitigation: keep artifacts on dedicated bind mount; keep code on fast path.
- Risk: Dual writers (host+container) reintroduce contention. Mitigation: guidance to avoid concurrent editing; save-all before agent runs.

## Rollout & Follow-up
- Steps: update devcontainer settings/docs; communicate extension guidance; apply mounts layout; validate with pytest benchmarks.
- Follow-up: monitor devcontainer pytest times; spot-check extension list in container; collect feedback from users.
- Links: issue #55, RCA, this spec, validation results (to be added).

