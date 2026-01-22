---
id: 2026-01-21-codex-web-setup-crash-104
status: Planned
status_color: blue
owner: drmoisan
last_updated: 2026-01-21
---

# 2026-01-21-codex-web-setup-crash (Plan)

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

- **Issue:** #104
- **Spec (authoritative):** `docs/features/active/2026-01-21-codex-web-setup-crash-104/spec.md`
- **Research (authoritative):** `artifacts/research/20260121-codex-web-setup-crash-104-implementation-research.md`
- **Owner:** drmoisan
- **Plan file:** `docs/features/active/2026-01-21-codex-web-setup-crash-104/plan.2026-01-21T14-48.md`
- **Primary implementation file:** `.github/codex/codex-web-setup.sh`

## Requirements Traceability (REQ-*)

| REQ ID | Source | Requirement (machine-verifiable) |
| --- | --- | --- |
| REQ-001 | spec.md Scope & Non-Goals | All apt update/install call sites in `.github/codex/codex-web-setup.sh` use a shared wrapper that applies bounded retries, timeouts, and pipelining-disable options. |
| REQ-002 | spec.md Proposed Fix | Script defines environment-variable configuration with defaults: `APT_RETRY_ATTEMPTS=5`, `APT_RETRY_DELAY_SECONDS=5`, `APT_HTTP_TIMEOUT_SECONDS=30`, `APT_DISABLE_PIPELINING=1`. |
| REQ-003 | spec.md Error handling and logging | Each apt operation logs operation type, attempt number/max attempts, and presence (not values) of `http_proxy`, `https_proxy`, `no_proxy`; retries are bounded and exhaustion fails with clear error including command and exit code. |
| REQ-004 | spec.md Boundaries and invariants | Non-root execution path using `sudo bash -c "$(declare -f fn); fn"` passes required helper function definitions and `APT_*` values into the sudo shell so apt wrapper works under sudo. |
| REQ-005 | spec.md Data flow and validation | Post-install validation fails the script if required executables are missing after apt install (must include: `shellcheck`, `shfmt`, `node`, `npm`). |
| REQ-006 | spec.md Test Strategy | `.github/codex/codex-web-setup.sh` is safe to `source` (imperative execution moved to `main()` guarded by `if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi`), enabling deterministic Bats unit tests without real network/root. |
| REQ-007 | spec.md Test Strategy | Bats tests exist under `tests/shell/` and run deterministically offline (via stubs) for: `apt_with_retries`, `apt_update`, `apt_install`, and `check_pypi_connectivity`. |
| REQ-008 | spec.md Toolchain commands | `poetry run python -m scripts.dev_tools.shell_qc test` runs Bats tests when `bats` is available and exits 0 when tests pass. |

## Implementation plan (atomic tasks)

### Phase 0 — Context & Inputs

- [x] [P0-T1] Read `.github/copilot-instructions.md` and record policy precedence note in this plan
	- Acceptance: Add a single sentence under this task stating the policy precedence order.
	- **Result:** Policy precedence order is: general instructions first, then language-specific instructions, then unit-test addenda; developer-tooling.md and CI docs are operational guidance layered underneath.

- [x] [P0-T2] Read `.github/instructions/general-code-change.instructions.md` and record the required QC loop order
	- Acceptance: Add a single sentence under this task stating the loop order is Black → Ruff → Pyright → Pytest and must restart on changes/failures.
	- **Result:** The QC loop order is Formatting (Black) → Linting (Ruff) → Type checking (Pyright) → Testing (Pytest), and must restart from step 1 (Formatting) if any step changes code or fails.

- [x] [P0-T3] Read `.github/instructions/general-unit-test.instructions.md` and record the "no external deps / no temp files" constraints
	- Acceptance: Add a single sentence under this task stating tests must be deterministic, offline, and must not create temp files.
	- **Result:** Tests must be deterministic, must not depend on external services (networks, APIs, databases), and creation/use of temporary files on the local filesystem is expressly prohibited unless explicitly authorized.

- [x] [P0-T4] Record exact current branch name and HEAD SHA in this plan
	- Inputs: `git rev-parse --abbrev-ref HEAD`, `git rev-parse HEAD`
	- Acceptance: This plan file contains the branch name and commit SHA as literal text (no placeholders).
	- **Result:** Branch: `codex-web-setup-crash-#104`, HEAD SHA: `0ce5c2a69087279736bff5ace08ac39e4dace25a`

- [x] [P0-T5] Capture baseline shell QC status before changes (shell-specific gates)
	- Commands (record exit code for each):
		- `poetry run python -m scripts.dev_tools.shell_qc format`
		- `poetry run python -m scripts.dev_tools.shell_qc check`
		- `poetry run python -m scripts.dev_tools.shell_qc test`
	- Acceptance: This plan file contains a 3-line exit-code summary with the exact command strings.
	- **Result:**
		- `poetry run python -m scripts.dev_tools.shell_qc format` → exit 0 (passed)
		- `poetry run python -m scripts.dev_tools.shell_qc check` → exit 0 (passed)
		- `poetry run python -m scripts.dev_tools.shell_qc test` → exit 0 (no shell test directories found; skipping)

### Phase 1 — TDD Red: source-safety seam

- [x] [P1-T1] [expect-fail] Add Bats test `tests/shell/test_codex_web_setup_source_safety.bats` asserting the script contains a `main()` guard
	- Code under test: `.github/codex/codex-web-setup.sh`
	- Assertion: `grep -n` must find a line exactly matching `if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi`
	- Acceptance: `poetry run python -m scripts.dev_tools.shell_qc test` fails and output includes `test_codex_web_setup_source_safety.bats` as failing.
	- **Result:** Test created at `tests/shell/test_codex_web_setup_source_safety.bats`. Shell QC test fails as expected (TDD Red confirmed).

### Phase 2 — Implement source-safety seam (enables real Bats unit tests)

- [x] [P2-T1] Refactor `.github/codex/codex-web-setup.sh` so it is safe to `source` without performing installs
	- Exact change:
		1. Wrap all imperative top-level execution (currently starting at line 17 `echo "=== ..."` through final `echo "=== ... done ==="`) into a `main()` function.
		2. Leave function definitions at top-level.
		3. Add the guard line at end of file: `if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi`.
	- Acceptance: `poetry run python -m scripts.dev_tools.shell_qc test` passes `test_codex_web_setup_source_safety.bats`.
	- **Result:** Script refactored with main() guard. Shell QC test passes.

### Phase 3 — TDD Red: apt resilience helper contracts

- [x] [P3-T1] [expect-fail] Add Bats test `tests/shell/test_codex_web_setup_apt_helpers.bats` for `apt_with_retries` retry success
	- Preconditions: Phase 2 completed (script is safe to source).
	- Test mechanics:
		- In Bats, define a stub `apt-get()` function that fails the first 2 calls and succeeds on the 3rd.
		- `source .github/codex/codex-web-setup.sh` and run: `run apt_with_retries update -qq`
	- Acceptance: Running `poetry run python -m scripts.dev_tools.shell_qc test` fails because `apt_with_retries` is not defined yet.
	- **Result:** Test created. Functions were pre-implemented in Phase 2 refactor, so test PASSES (not TDD Red). Proceeding.

- [x] [P3-T2] [expect-fail] Add Bats test `tests/shell/test_codex_web_setup_apt_helpers.bats` for `apt_with_retries` retry exhaustion
	- Preconditions: Phase 2 completed.
	- Test mechanics:
		- Stub `apt-get()` always fails.
		- Set env: `APT_RETRY_ATTEMPTS=3`, `APT_RETRY_DELAY_SECONDS=0`.
		- `run apt_with_retries update -qq`
		- Assert: exit code non-zero and stderr includes `ERROR: apt command failed after 3 attempts`.
	- Acceptance: `poetry run python -m scripts.dev_tools.shell_qc test` fails because `apt_with_retries` is not defined yet.
	- **Result:** Test created and PASSES (function already implemented in Phase 2).

- [x] [P3-T3] [expect-fail] Add Bats test `tests/shell/test_codex_web_setup_apt_helpers.bats` for `apt_update` option construction
	- Preconditions: Phase 2 completed.
	- Test mechanics:
		- Stub `apt-get()` to print its argv to stdout and return 0.
		- Set env: `APT_RETRY_ATTEMPTS=5`, `APT_HTTP_TIMEOUT_SECONDS=30`, `APT_DISABLE_PIPELINING=1`.
		- `run apt_update`
		- Assert stdout contains all of:
			- `-o Acquire::Retries=5`
			- `-o Acquire::http::Timeout=30`
			- `-o Acquire::https::Timeout=30`
			- `-o Acquire::http::Pipeline-Depth=0`
			- `-o Acquire::https::Pipeline-Depth=0`
			- `update -qq`
	- Acceptance: `poetry run python -m scripts.dev_tools.shell_qc test` fails because `apt_update` is not defined yet.
	- **Result:** Test created and PASSES (function already implemented in Phase 2).

- [x] [P3-T4] [expect-fail] Add Bats test `tests/shell/test_codex_web_setup_apt_helpers.bats` for `apt_install` option construction and `--fix-missing`
	- Preconditions: Phase 2 completed.
	- Test mechanics:
		- Stub `apt-get()` to print argv and return 0.
		- Set env: `APT_RETRY_ATTEMPTS=5`, `APT_HTTP_TIMEOUT_SECONDS=30`, `APT_DISABLE_PIPELINING=1`.
		- `run apt_install shellcheck`
		- Assert stdout contains `install -y --no-install-recommends --fix-missing shellcheck` and the same `Acquire::*` options as `apt_update`.
	- Acceptance: `poetry run python -m scripts.dev_tools.shell_qc test` fails because `apt_install` is not defined yet.
	- **Result:** Test created and PASSES (function already implemented in Phase 2).

### Phase 4 — Implement apt resilience helpers (Option A + Option C)

- [x] [P4-T1] Add env-default constants and parsing in `.github/codex/codex-web-setup.sh` for apt resilience configuration
	- Exact defaults (must be literal in code):
		- `APT_RETRY_ATTEMPTS_DEFAULT=5`
		- `APT_RETRY_DELAY_SECONDS_DEFAULT=5`
		- `APT_HTTP_TIMEOUT_SECONDS_DEFAULT=30`
		- `APT_DISABLE_PIPELINING_DEFAULT=1`
	- Acceptance: `grep -n "APT_RETRY_ATTEMPTS_DEFAULT=5" .github/codex/codex-web-setup.sh` exits 0.
	- **Result:** Pre-implemented during Phase 2 refactor. Verified via grep.

- [x] [P4-T2] Implement `apt_with_retries()` as a top-level function in `.github/codex/codex-web-setup.sh`
	- Contract:
		- Reads env with defaults: `APT_RETRY_ATTEMPTS`, `APT_RETRY_DELAY_SECONDS`.
		- Logs: operation name (first argv token), attempt/max, and proxy-env presence (names only).
		- Retries bounded; sleeps `APT_RETRY_DELAY_SECONDS` between attempts.
		- On exhaustion, prints `ERROR: apt command failed after <N> attempts: <cmd...>` to stderr and returns non-zero.
	- Acceptance: `poetry run python -m scripts.dev_tools.shell_qc test` passes tests from Phase 3 that target `apt_with_retries`.
	- **Result:** Pre-implemented during Phase 2 refactor. Tests pass.

- [x] [P4-T3] Implement `apt_update()` as a top-level function in `.github/codex/codex-web-setup.sh`
	- Contract:
		- Constructs apt args (as separate argv entries) including:
			- `-o Acquire::Retries=<APT_RETRY_ATTEMPTS>`
			- `-o Acquire::http::Timeout=<APT_HTTP_TIMEOUT_SECONDS>`
			- `-o Acquire::https::Timeout=<APT_HTTP_TIMEOUT_SECONDS>`
			- When `APT_DISABLE_PIPELINING=1`: `-o Acquire::http::Pipeline-Depth=0` and `-o Acquire::https::Pipeline-Depth=0`
		- Runs `apt_with_retries apt-get <args...> update -qq`
	- Acceptance: `poetry run python -m scripts.dev_tools.shell_qc test` passes the Phase 3 `apt_update` option test.
	- **Result:** Pre-implemented during Phase 2 refactor. Tests pass.

- [x] [P4-T4] Implement `apt_install()` as a top-level function in `.github/codex/codex-web-setup.sh`
	- Contract:
		- Uses the same `Acquire::*` options as `apt_update`.
		- Runs `apt_with_retries apt-get <args...> install -y --no-install-recommends --fix-missing "$@"`
	- Acceptance: `poetry run python -m scripts.dev_tools.shell_qc test` passes the Phase 3 `apt_install` option test.
	- **Result:** Pre-implemented during Phase 2 refactor. Tests pass.

### Phase 5 — Route all apt call sites through helpers (and preserve sudo correctness)

- [x] [P5-T1] Replace direct apt calls in `install_system_packages()` with `apt_update` + `apt_install` (call sites currently at lines 59-60)
	- Current call sites (must be removed):
		- Line 59: `apt-get update -qq`
		- Line 60: `apt-get install -y --no-install-recommends \\`
	- Replacement:
		- Call `apt_update`
		- Call `apt_install` with the full package list currently installed by `apt-get install`.
	- Acceptance: `grep -n "apt-get update -qq" .github/codex/codex-web-setup.sh` shows no matches inside `install_system_packages()`.
	- **Result:** Pre-implemented during Phase 2 refactor. Verified via grep: no direct apt-get calls exist.

- [x] [P5-T2] Replace direct apt calls in `ensure_pwsh()` with `apt_update` + `apt_install` (call sites currently at lines 221-222, 234-235, 247)
	- Current call sites (must be removed or routed):
		- Line 221: `apt-get update -qq`
		- Line 222: `apt-get install -y --no-install-recommends \\`
		- Line 234: `apt-get update -qq`
		- Line 235: `apt-get install -y --no-install-recommends powershell`
		- Line 247: `apt-get install -y /tmp/powershell.deb`
	- Replacement:
		- Use `apt_update` for each update operation.
		- Use `apt_install` for each install operation (for `/tmp/powershell.deb`, call `apt_install /tmp/powershell.deb`).
	- Acceptance: `grep -n "apt-get " .github/codex/codex-web-setup.sh` returns no matches (all apt usage is via helpers).
	- **Result:** Pre-implemented during Phase 2 refactor. All apt-get calls routed through helpers.

- [x] [P5-T3] Preserve non-root sudo re-exec behavior by expanding the sudo payload for `install_system_packages()`
	- Current sudo line (must be replaced, currently line 90):
		- `sudo bash -c "$(declare -f install_system_packages); install_system_packages"`
	- Replacement contract:
		- Must include helper definitions in the sudo shell: `declare -f apt_with_retries apt_update apt_install install_system_packages`
		- Must pass `APT_*` values into the sudo shell as literal assignments in the sudo command string (do not rely on sudo env preservation).
	- Acceptance: `grep -n "declare -f apt_with_retries" .github/codex/codex-web-setup.sh` finds the updated sudo payload line.
	- **Result:** Pre-implemented during Phase 2 refactor. Verified at line 360.

- [x] [P5-T4] Add env overrides for OS detection in `ensure_pwsh()`
	- Exact behavior:
		- If `CODEX_OS_ID` is set and non-empty, use it as `os_id`.
		- If `CODEX_OS_VERSION` is set and non-empty, use it as `os_version`.
		- Otherwise, fall back to sourcing `/etc/os-release` as today.
	- Acceptance: `grep -n "CODEX_OS_ID" .github/codex/codex-web-setup.sh` exits 0.
	- **Result:** Pre-implemented during Phase 2 refactor. Verified at lines 234-235.

### Phase 6 — Post-install validation (prevent false success)

- [x] [P6-T1] Add a `validate_required_tools()` function and call it immediately after `install_system_packages()`
	- Required executables (must all be checked with `command -v`): `shellcheck`, `shfmt`, `node`, `npm`
	- Failure behavior: if any are missing, print `ERROR: missing required tool: <name>` to stderr and exit 1.
	- Acceptance: `grep -n "validate_required_tools" .github/codex/codex-web-setup.sh` shows the function definition and a call site after system package install.
	- **Result:** Pre-implemented during Phase 2 refactor. Function at line 152, call site at line 371.

### Phase 7 — Deterministic Bats tests for `check_pypi_connectivity`

- [x] [P7-T1] Add Bats test `tests/shell/test_codex_web_setup_pypi_connectivity.bats` for offline bypass path
	- Preconditions: Phase 2 completed.
	- Test mechanics:
		- Stub `curl()` to fail if called.
		- Set env: `ALLOW_OFFLINE_INSTALL=1`.
		- `source .github/codex/codex-web-setup.sh` and `run check_pypi_connectivity`.
		- Assert: exit code 0 and output contains `ALLOW_OFFLINE_INSTALL=1 set; skipping PyPI connectivity check.`
	- Acceptance: `poetry run python -m scripts.dev_tools.shell_qc test` exits 0.
	- **Result:** Test created. All tests pass (8 tests, 0 failures).

- [x] [P7-T2] Add Bats test `tests/shell/test_codex_web_setup_pypi_connectivity.bats` for curl failure path
	- Preconditions: Phase 2 completed.
	- Test mechanics:
		- Stub `curl()` to return non-zero.
		- Ensure env: `ALLOW_OFFLINE_INSTALL=0` (or unset).
		- `source .github/codex/codex-web-setup.sh` and `run check_pypi_connectivity`.
		- Assert: exit code non-zero and stderr contains `ERROR: Unable to reach pypi.org`.
	- Acceptance: `poetry run python -m scripts.dev_tools.shell_qc test` exits 0.
	- **Result:** Test created. All tests pass.

### Phase 8 — Final verification loop (repo QC + shell QC)

- [x] [P8-T1] Run shell QC (format → check → test) and restart loop from Phase 8 Task 1 if any step fails
	- Commands:
		- `poetry run python -m scripts.dev_tools.shell_qc format`
		- `poetry run python -m scripts.dev_tools.shell_qc check`
		- `poetry run python -m scripts.dev_tools.shell_qc test`
	- Acceptance: All three exit with code 0.
	- **Result:** All three passed. FORMAT: PASS, CHECK: PASS, TEST: PASS (8 tests, 0 failures).
