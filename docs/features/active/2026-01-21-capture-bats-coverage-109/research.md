<!-- markdownlint-disable-file -->

# Task Research Notes: capture-bats-coverage (#109)

## Research Executed

### File Analysis

- `/workspaces/lexile-corpus-tuner/docs/features/active/2026-01-21-capture-bats-coverage-109/issue.md`
  - Confirms the goal is a Bash coverage workflow comparable to PowerShell/Python, with deterministic artifacts under `artifacts/`, HTML + machine-readable summary, CI compatibility, and a staged rollout (reporting first).

- `/workspaces/lexile-corpus-tuner/artifacts/chats/2026-01-21T22-22 Measuring Coverage in BATS Testing.md`
  - Captures the key premise: `bats-core` does not provide coverage; a separate instrumentation tool is required.
  - Proposes `kcov` + VS Code Coverage Gutters as the closest analogue to PowerShell `koverage` + Coverage Gutters.

- `/workspaces/lexile-corpus-tuner/scripts/dev_tools/shell_qc.py`
  - Current Shell QC supports `check` (shfmt -d + shellcheck), `format` (shfmt -w), and `test` (bats), but has **no coverage flag**.
  - Shell tests run only if `tests/shell` or `tests/bash` exist (relative to repo root).
  - Tool execution uses `subprocess.run(["shfmt", ...])` / `subprocess.run(["shellcheck", ...])` / `subprocess.run(["bats", ...])`.

- `/workspaces/lexile-corpus-tuner/tests/scripts/dev_tools/test_shell_qc.py`
  - Unit tests validate the current Shell QC surface.
  - Confirms the discovered Bats test directory in fixtures (not necessarily in the repo root), and verifies the “skipping” behaviors.

- `/workspaces/lexile-corpus-tuner/.vscode/settings.json`
  - Coverage Gutters is configured to look for coverage files under `coverage-gutters.coverageBaseDir: "artifacts/pester/**"`.
  - `coverage-gutters.coverageFileNames` includes `"cov.xml"`, `"cobertura.xml"`, and `"powershell-coverage.xml"`.
  - This indicates the repo already anticipates multiple formats and multiple reports; the missing piece is generating Bash coverage into a discoverable location.

- `/workspaces/lexile-corpus-tuner/.devcontainer/Dockerfile`
  - Installs `shellcheck` and `shfmt` and builds `bashdb`, but does **not** install `kcov`.

- `/workspaces/lexile-corpus-tuner/docs/developer-tooling.md`
  - Documents Shell QC commands (`shell_qc format/check/test`) but no coverage workflow for Bash.

### Code Search Results

- `shell_qc|shell-qc|kcov|bats|Coverage Gutters|coverage-gutters|coverageGutters|cobertura`
  - Found references in the active feature folder, policy audit templates, and the chat artifact.
  - Found that policy audit templates currently treat Bash coverage as “N/A” (indicating a deliberate gap today).

### External Research

- #fetch:https://github.com/drmoisan/lexile-corpus-tuner/issues/109
  - Confirms the GitHub issue content matches `issue.md` (the promoted potential).

- #fetch:https://github.com/SimonKagstrom/kcov/raw/refs/heads/master/doc/vscode.md
  - `kcov --cobertura-only ...` produces `cov.xml` output suitable for VS Code Coverage Gutters.
  - Recommends limiting Coverage Gutters search scope (performance), i.e., configure it to look in only the output dir.

- #fetch:https://raw.githubusercontent.com/SimonKagstrom/kcov/master/README.md
  - kcov supports Bash coverage and outputs:
    - HTML output
    - Cobertura-compatible XML
    - JSON summary
  - Supports filtering via `--include-pattern` / `--exclude-pattern` or path-based `--include-path` / `--exclude-path`.
  - Supports merging multiple runs via `kcov --merge <out> <run1> <run2> ...`.

- #fetch:https://raw.githubusercontent.com/SimonKagstrom/kcov/master/INSTALL.md
  - Ubuntu/Debian build dependencies include (verbatim from upstream):
    `binutils-dev build-essential cmake libssl-dev libcurl4-openssl-dev libelf-dev libstdc++-12-dev zlib1g-dev libdw-dev libiberty-dev`.

- #fetch:https://raw.githubusercontent.com/SimonKagstrom/kcov/master/doc/docker.md
  - kcov publishes a docker image `kcov/kcov` and shows how to copy `kcov*` into your own image.
  - Notes kcov requires system calls like `ptrace` and may need `--security-opt seccomp=unconfined` in some docker contexts.
  - Provides an example that installs runtime dependencies + bats and runs: `kcov --include-path=/code --dump-summary ./coverage bats ./test.sh`.

- #fetch:https://raw.githubusercontent.com/ryanluker/vscode-coverage-gutters/master/package.json
  - Coverage Gutters setting `coverage-gutters.coverageBaseDir` explicitly supports a **workspaceFolder-relative path or glob** and uses it to resolve coverage files as:
    `${coverageBaseDir}/{${coverageFileNames}}`.
  - Confirms multi-file support and the default names include `cov.xml`.

### Project Conventions

- Standards referenced: repo uses “toolchain loop” conventions (format → lint → type-check → test), and has per-language ecosystems (Python + PowerShell) documented in `docs/developer-tooling.md`.
- Instructions followed: Task Researcher Instructions (research-only; write results to `artifacts/research/`).

## Key Discoveries

### Project Structure

- Shell tooling is centralized in `scripts/dev_tools/shell_qc.py` and exposed through Poetry entry points (`poetry run shell-qc ...`).
- Shell tests are expected under `tests/shell` or `tests/bash`, but **these directories do not exist in the repo root today** (verified by listing `/workspaces/lexile-corpus-tuner/tests` and observing no `shell/` or `bash/` children).
  - Implication: a Bash coverage workflow will likely need to introduce at least one Bats test directory in-repo (or wire to an existing location if one exists elsewhere).

### Local Environment Verification

- Bats is installed in the devcontainer runtime:
  - `bats --version` => `Bats 1.8.2`
- kcov is not currently installed:
  - `kcov --version` => NOT INSTALLED (binary not found on PATH)
- Repo root test directories confirm Shell QC will skip by default today:
  - `Get-ChildItem /workspaces/lexile-corpus-tuner/tests -Directory` => `__pycache__`, `fixtures`, `lexile_scoring_model`, `scripts`, `src`

### Implementation Patterns

- The repo already uses Coverage Gutters and already includes `cov.xml` in `coverage-gutters.coverageFileNames`.
  - This aligns cleanly with the upstream kcov + `--cobertura-only` recommendation.
- The devcontainer installs shell tooling via apt and also builds tools from source when necessary (example: `bashdb`).
  - This is relevant because `kcov` does not appear to be available via apt in this container by default.

### Complete Examples

```bash
# kcov upstream pattern for VS Code Coverage Gutters
# Source: https://github.com/SimonKagstrom/kcov/raw/refs/heads/master/doc/vscode.md
kcov --cobertura-only [other options] /path/to/your/folder/.vscode /path/to/the/binary

# kcov upstream filtering examples (pattern-based)
# Source: https://raw.githubusercontent.com/SimonKagstrom/kcov/master/README.md
kcov --exclude-pattern=/usr/include --include-pattern=part/of/path,other/path \
  /path/to/outdir executable

# kcov upstream merging example
# Source: https://raw.githubusercontent.com/SimonKagstrom/kcov/master/README.md
kcov --merge /tmp/merged-output /tmp/kcov-output1 /tmp/kcov-output2

# docker-based example (kcov + bats)
# Source: https://raw.githubusercontent.com/SimonKagstrom/kcov/master/doc/docker.md
kcov --include-path=/code --dump-summary ./coverage bats ./test.sh
```

### API and Schema Documentation

- kcov produces `cov.xml` under `--cobertura-only`, which Coverage Gutters can consume.
- Coverage Gutters resolves coverage files via:
  - `coverage-gutters.coverageBaseDir` (workspace-relative path or glob)
  - and `coverage-gutters.coverageFileNames` (array of names, includes `cov.xml`)

### Configuration Examples

```json
// Source: repo file /workspaces/lexile-corpus-tuner/.vscode/settings.json
{
  "coverage-gutters.coverageBaseDir": "artifacts/pester/**",
  "coverage-gutters.coverageFileNames": [
    "lcov.info",
    "cov.xml",
    "coverage.xml",
    "cobertura.xml",
    "jacoco.xml",
    "coverage.cobertura.xml",
    "powershell-coverage.xml"
  ]
}
```

### Technical Requirements

- `bats-core` does not provide coverage; coverage requires an external tool.
- `kcov` is a strong candidate because:
  - It natively supports Bash.
  - It outputs both HTML + Cobertura XML + JSON.
  - It has explicit documentation for VS Code Coverage Gutters.
- If using kcov in containerized contexts, runtime permissions may matter (ptrace/seccomp notes in upstream docker docs).

**Mandatory unachievable objective callout**:
- **Installing `kcov` via `apt-get install kcov` is not currently achievable in this devcontainer base as configured.** Evidence:
  - `apt-cache search kcov` produced no results.
  - `apt-cache show kcov` => `E: No packages found`.
  A non-apt installation path (build from source or copy from `kcov/kcov` image) is therefore required.

## Recommended Approach

Use `kcov` + Coverage Gutters, with kcov installed in the devcontainer via a non-apt route.

Why this is the best fit (evidence-based):

- Upstream kcov explicitly documents how to generate `cov.xml` for Coverage Gutters (`--cobertura-only`).
- The repo already configures Coverage Gutters to search under `artifacts/pester/**` and includes `cov.xml`.
- kcov supports include/exclude filters and merging multiple runs (needed for “ecosystem-like” workflows).

Installation strategy (ranked):

1. **Preferred: multi-stage copy from the official `kcov/kcov` docker image** into the devcontainer image.
   - Evidence: upstream `doc/docker.md` provides a canonical `COPY --from=kcov/kcov:latest /usr/local/bin/kcov* /usr/local/bin/` pattern.
   - Benefit: avoids compiling during devcontainer builds (faster and less fragile).
   - Risk: still requires runtime dependencies and ptrace permissions.

2. **Fallback: build kcov from source during devcontainer build**.
   - Evidence: upstream `INSTALL.md` lists Ubuntu/Debian build dependencies.
   - Benefit: does not depend on docker multi-stage supply-chain, but increases build time.

Once installed:

- Add a new Shell QC coverage path (e.g., `shell-qc test --coverage`) that:
  - Runs Bats under kcov
  - Emits `cov.xml` + HTML report under a stable directory within `artifacts/` (so Coverage Gutters can display it).
  - Uses kcov include/exclude rules to limit coverage to repo-owned shell scripts.

## Implementation Guidance

- **Objectives**: Provide a Bash coverage workflow (Bats-driven) that is visible in VS Code via Coverage Gutters and emits deterministic artifacts suitable for CI.
- **Key Tasks**:
  - Install kcov in the devcontainer via docker multi-stage copy or source build.
  - Add a Shell QC “coverage” mode producing `cov.xml`.
  - Ensure coverage output path matches Coverage Gutters configuration (or update config accordingly).
  - Add at least one in-repo `tests/shell` (or `tests/bash`) Bats test to validate end-to-end behavior.
  - Document the workflow in `docs/developer-tooling.md`.
- **Dependencies**:
  - kcov runtime dependencies (per upstream docker docs) and/or build dependencies (per `INSTALL.md`).
  - bats (if we want Shell QC tests in CI and coverage runs by default).
- **Success Criteria**:
  - Running the coverage command generates:
    - a `cov.xml` discoverable by Coverage Gutters
    - an HTML report directory under `artifacts/`
    - a summary (kcov supports JSON/summary output; exact format to be chosen during implementation)
  - CI can run the coverage command without interactive prompts.
