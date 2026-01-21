# 2026-01-21-codex-web-setup-crash (Spec)

- **Issue:** #104
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-01-21T14-48
- **Status:** Draft
- **Version:** 0.1

## Context
Codex web environment setup fails while installing system packages.
The bootstrap script exits with code 100 after an apt mirror returns a 502.

Environment:
- OS/version: Linux (Codex web job; Ubuntu noble mirror in apt output)
- Python version: Unknown
- Command/flags used: `.github/codex/codex-web-setup.sh`
- Data source or fixture: None

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Run a Codex web job that executes `.github/codex/codex-web-setup.sh`.
2. Wait for the system package install step.
3. Observe apt failing during package fetch.

Expected:
The setup script completes and the environment is fully provisioned.

Actual:
The script exits with code 100 during apt install.
Key error: `Failed to fetch ... 502 Bad Gateway` from the Ubuntu mirror.

Logs / Screenshots:
- [x] Attached minimal logs or screenshot
- Snippet:
	```
	E: Failed to fetch http://archive.ubuntu.com/ubuntu/pool/universe/n/node-cjs-module-lexer/node-cjs-module-lexer_1.2.3%2bdfsg-1_all.deb  502  Bad Gateway [IP: 172.31.1.19 8080]
	E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
	```


## Scope & Non-Goals
- In scope:
	- Harden `.github/codex/codex-web-setup.sh` against transient apt transport failures in Codex web jobs.
	- Apply the hardening consistently to all apt update/install call sites in the script.
	- Add lightweight diagnostic logging that helps distinguish proxy/gateway problems from mirror or package issues.
	- Preserve correctness in both execution modes:
		- running as root, and
		- running as non-root using the script’s `sudo bash -c "$(declare -f fn); fn"` re-exec pattern.
	- Add deterministic, offline unit tests for the new bash helper functions using Bats.

- Out of scope / non-goals:
	- Changing Codex runner infrastructure, proxy configuration, or Ubuntu mirror selection (we can only react to these from within the script).
	- Replacing the overall toolchain being provisioned (package list should remain aligned with the devcontainer parity intent unless explicitly approved in a follow-up).
	- Adding a new external caching proxy service (e.g., apt-cacher-ng) or pinning a custom mirror.

- Explicitly excluded systems, integrations, or datasets:
- No corpus data or external datasets involved.
- No changes to the Python package or CLI behavior.

## Root Cause Analysis
Evidence indicates the Codex web job routes apt traffic through an HTTP proxy/gateway:

- Apt error includes `502  Bad Gateway [IP: 172.31.1.19 8080]` while fetching from `http://archive.ubuntu.com/...`.
- The install downloads many packages successfully (e.g., `Fetched 30.7 MB in 56s`) and then fails on a single artifact fetch (`node-cjs-module-lexer_...deb`).

Most likely cause (for this repo): transient proxy/gateway instability or proxy edge-case behavior under load, not a deterministic bug in the repo.


## Proposed Fix

### Design summary (what changes where):

- Add a small apt wrapper inside `.github/codex/codex-web-setup.sh` that applies:
	- Option A: bounded retries + timeouts for apt operations
	- Option C: disable HTTP pipelining to reduce proxy sensitivity
- Use the wrapper for all apt `update` and `install` operations, including PowerShell install paths.

Design note (based on current workspace state):

- Option A (bounded retries + timeouts) is not yet implemented in the script.
	- Today the script calls `apt-get update -qq` and `apt-get install ...` directly.
	- The fix will introduce a small wrapper (e.g., `apt_with_retries`, plus `apt_update`/`apt_install`) so
		all apt operations use consistent retry/timeout behavior.
- Option C (HTTP pipelining disable) is not yet represented in the script; it should be added explicitly
	(e.g., `Acquire::http::Pipeline-Depth=0` and `Acquire::https::Pipeline-Depth=0`).

### Boundaries and invariants to preserve:

- Preserve the script’s overall purpose: bootstrap a Codex web job environment with devcontainer-like tooling.
- Preserve the existing package list intent (do not drop or replace packages as a “fix”).
- Preserve compatibility with both:
	- running as root, and
	- running as non-root with the script’s `sudo bash -c "$(declare -f fn); fn"` re-exec pattern.
- Do not allow a “false success” state where the script exits 0 but required tools are missing.

Invariant detail (important once the wrapper is introduced):

- When executing via the non-root sudo re-exec pattern, the sudo shell must receive all helper function
	definitions required by the invoked function (e.g., the planned `apt_with_retries`, `apt_update`,
	`apt_install`) and the `APT_*` variables.

### Dependencies or blocked work:

- None. This is implemented entirely within `.github/codex/codex-web-setup.sh` using documented apt options.

### Implementation strategy (what changes, not sequencing):
	
#### Files/modules to change:

- `.github/codex/codex-web-setup.sh`

#### Functions/classes/CLI commands impacted:

- `install_system_packages()`
	- Change: route `apt-get update` and `apt-get install` through the new apt wrapper.
- `ensure_pwsh()`
	- Change: route all apt update/install calls (and any fallback install prerequisites) through the new apt wrapper.
- No new CLI commands; script entry point remains the same.

#### Data flow and validation changes:

- No data flow changes.
- Add a post-install validation step for “must-have” executables (examples: `shellcheck`, `shfmt`, `node`, `npm`).
	- Rationale: prevents success if a retry path uses `--fix-missing` or a partial install occurs.
	- Note: this is especially important because the current apt wrapper uses `--fix-missing`.

#### Error handling and logging updates:

- For each apt operation, log:
	- the operation type (`update` vs `install`),
	- attempt number / max attempts,
	- whether proxy-related environment variables are set (`http_proxy`, `https_proxy`, `no_proxy`) without printing their values.
- If retries are exhausted, fail with a clear error that includes the failing apt command and exit code.

#### Rollback/feature-flag considerations (if applicable):

- Make behavior configurable via environment variables with safe defaults:
	- (Planned) `APT_RETRY_ATTEMPTS` (default: 5)
	- (Planned) `APT_RETRY_DELAY_SECONDS` (default: 5)
	- (Planned) `APT_HTTP_TIMEOUT_SECONDS` (default: 30)
	- (Planned) `APT_DISABLE_PIPELINING` (default: 1; when enabled sets `Acquire::http::Pipeline-Depth=0`)
- Rollback is achieved by setting:
	- `APT_RETRY_ATTEMPTS=1` and/or `APT_DISABLE_PIPELINING=0` to revert to near-current behavior.

### Technical specifications (interfaces/contracts):

#### Inputs/outputs and formats:

- Inputs:
	- Existing script arguments/behavior (no change).
	- Optional environment variables listed above (strings/ints).
- Outputs:
	- Standard output logs indicating apt retry behavior and proxy-env presence.
	- Non-zero exit on failure (current behavior preserved).

#### Required configuration keys and defaults:

- No repo config files are required.
- Defaults are provided in the script via environment-variable fallbacks.

#### Backward-compatibility expectations:

- Script remains runnable in environments without proxy env vars.
- Script remains runnable when executed as root or non-root (sudo re-exec path).
- No changes to project Python dependencies or CLI entry points.

#### Performance constraints (latency/throughput/memory):

- Retries must be bounded; the script should not retry indefinitely.
- Disabling pipelining may slow down apt fetches; this is acceptable for reliability in Codex web jobs.
- Retry delay should be small but non-zero to reduce immediate retry against a struggling proxy.

## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
	- Codex web jobs may run behind an HTTP proxy/gateway that can intermittently fail.
	- The job environment uses Ubuntu apt sources (`archive.ubuntu.com`, Ubuntu noble) during setup.

- Constraints (budget, performance, compatibility):
	- Must remain a single-script change (no new services).
	- Must preserve current script semantics (fail fast after bounded retries; no silent partial installs).

- External dependencies (services, libraries, releases):
	- Ubuntu apt mirrors and any upstream proxy/gateway in the Codex environment.

Research sufficiency:
	- The provided research artifact is sufficient to complete this spec (root-cause hypothesis, apt knobs,
		and concrete mitigation options are evidence-backed).

## Data / API / Config Impact
- User-facing or API changes:
	- None.

- Data or migration considerations:
	- None.

- Logging/telemetry updates (if any):
	- Add setup-script logs for apt retry attempts and proxy-env presence (values redacted).

- Compatibility notes (CLI flags, config schemas, versioning):
	- No CLI/schema changes.

## Test Strategy
Seeded from issue:

- [x] Unit coverage areas
- [x] Integration scenario to retest
- [x] Manual verification notes

Integration scenario to retest: rerun the Codex web setup job after adding apt retries/timeout handling.
Manual verification notes: confirm the script completes and `shell-qc` tools install successfully.

- Regression tests to add or update:
	- Primary regression is the Codex web job rerun (environment-specific proxy behavior cannot be deterministically reproduced offline).
	- Add Bats unit tests under `tests/shell/` (or `tests/bash/`) once the setup script is safe to source (see below).

- Unit tests (Bats) for the fixed behavior and boundaries:
	- Add deterministic tests for the bash helper functions introduced by this change:
		- (Planned) `apt_with_retries` (retry success + retry exhaustion)
		- (Planned) `apt_update` (correct `apt-get` option construction)
		- (Planned) `apt_install` (correct `apt-get` option construction, includes `--fix-missing`)
		- `check_pypi_connectivity` (`ALLOW_OFFLINE_INSTALL=1` bypass; curl failure path)
	- Implementation requirement for unit tests:
		- Refactor `.github/codex/codex-web-setup.sh` to be safe to `source` by moving imperative execution into
			a `main()` function guarded by `if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi`.
		- Rationale: Bats should be able to stub external commands (e.g., `apt-get`, `curl`, `wget`, `dpkg`, `sleep`) in-process
			without executing real installs.

- Edge cases and negative scenarios (invalid inputs, missing data, boundary values):
	- Proxy env vars present vs absent.
	- Retries exhausted (script should fail with clear error).
	- Partial install risk (post-install required-tool validation must fail if tools are missing).
	- Non-root execution path uses sudo re-exec; helper functions must be available in the sudo shell.

- Error handling and logging verification:
	- Verify logs include: operation type, attempt counts, and whether proxy env vars are set (without printing proxy secrets).
	- Verify failures include the failing URL/command line from apt output.

- Coverage impact and targets for changed lines/modules:
	- N/A (bash script).

- Toolchain commands to run (format → lint → type-check → test):
	- Bash QC:
		- `poetry run python -m scripts.dev_tools.shell_qc format`
		- `poetry run python -m scripts.dev_tools.shell_qc check`
		- `poetry run python -m scripts.dev_tools.shell_qc test` (may skip if no tests; should still exit 0)

- Manual validation steps (if required):
	- Run the Codex web job at least once post-fix and confirm:
		- the setup completes,
		- required tools are present (`shellcheck`, `shfmt`, `node`, `npm`),
		- the logs clearly show retry attempts when failures occur.


## Acceptance Criteria
- [ ] Repro steps now produce the expected behavior in Codex web jobs: setup completes without exiting on a single transient 502.
- [ ] If apt fetch failures occur, the script retries boundedly and either:
	- succeeds, or
	- fails with a clear message after exhausting retries (no silent success).
- [ ] The script logs include attempt counts and proxy-env presence (values redacted).
- [ ] Post-setup validation confirms required tools are installed: `shellcheck`, `shfmt`, `node`, `npm`.
- [ ] No unintended behavior changes outside the defined scope (package list intent preserved; PowerShell bootstrap still works).
- [ ] Performance constraints met: retries are bounded and configurable via env vars.
- [ ] Bash QC pass completed: `shell_qc format` → `shell_qc check` → `shell_qc test`.
- [ ] Non-root execution remains correct: the sudo re-exec path can successfully call the apt wrappers (no missing-function errors).
- [ ] Bats unit tests are added for the apt wrapper helpers and run under `shell_qc test`.
- [ ] This spec reflects the chosen mitigation (A + C) and links to the supporting research artifact.

## Risks & Mitigations
- Technical or operational risks:
	- See detailed list below.

- Mitigations and rollbacks:
	- See detailed list below.

- Technical or operational risks:
	- Retries may increase setup duration when the proxy is unhealthy.
	- Disabling HTTP pipelining may reduce download performance.
	- If the proxy/gateway is consistently failing (deterministic outage/policy), retries will not help.
	- `--fix-missing` can mask a partial install unless we validate required tools.
	- The sudo re-exec path can regress if helper functions are not included in the sudo payload.

- Mitigations and rollbacks:
	- Keep retries bounded and configurable (env vars).
	- Add post-install required-tool checks to prevent false successes.
	- Provide rollback knobs to disable pipelining change and/or reduce retries to 1.
	- Make the sudo path explicit and self-contained (either include helper functions in `declare -f` or
		re-exec the whole script under sudo).

## Rollout & Follow-up
- Release/rollout steps:
	- Merge the change to `.github/codex/codex-web-setup.sh`.
	- Rerun the Codex web job and attach the successful run output (or failure output with improved diagnostics)
		to Issue #104.
	- Run repo Bash QC locally: `shell_qc format` → `shell_qc check` → `shell_qc test`.
	- If Bats tests are added, ensure the Codex job environment includes `bats` (or document expected skip
		behavior) so the new tests are actually exercised.

- Post-fix monitoring or clean-up tasks:
	- If failures persist, revisit proxy-scoped mitigations (e.g., additional apt proxy configuration) in a follow-up issue.
	- If failures persist specifically during HTTP transport, consider adding (or making configurable) the
		`Acquire::http::Pipeline-Depth=0` and `Acquire::https::Pipeline-Depth=0` options (Option C).

- Links: issue, PRs, related docs
	- Issue: #104
	- Research: `artifacts/research/20260121-codex-web-setup-crash-104-implementation-research.md`

- Release/rollout steps:
	- (See above.)

- Post-fix monitoring or clean-up tasks:
	- (See above.)

- Links: issue, PRs, related docs
	- (See above.)
