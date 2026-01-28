# Developer Tooling

This document describes the development tools, workflows, and setup procedures for the Lexile Corpus Tuner project.

## Development Environment Setup

### Python Version

- Python 3.10+ required
- Recommended: Python 3.13 for best performance and type checking

### Package Manager

- **Poetry** is the primary package manager
- Install via: `pip install poetry` or follow [Poetry installation guide](https://python-poetry.org/docs/#installation)

### Installation

**Standard installation:**

```bash
poetry install
```

**With optional dependencies:**

```bash
# For Lexile v2 model support
poetry install -E lexile-v2

# For OpenAI LLM rewriting
poetry install -E llm-openai

# All extras
poetry install --all-extras
```

**Legacy pip installation:**

```bash
pip install -e .                    # Basic install
pip install -e .[lexile-v2]        # With Lexile v2
pip install -e .[llm-openai]       # With OpenAI
```

## Code Quality Tools

### Formatting: Black

- **Configuration**: `pyproject.toml` `[tool.black]`
- **Line length**: 88 characters
- **Target**: Python 3.12

**Run formatter:**

```bash
poetry run black .
```
Prefer the command above instead of the legacy `QC: 1 Black: format` task.

### Linting: Ruff

- **Configuration**: `pyproject.toml` `[tool.ruff]`
- **Rules**: See project configuration for enabled rule sets

**Run linter:**

```bash
poetry run ruff check
```

**Auto-fix issues:**

```bash
poetry run ruff check --fix
```
Prefer these commands instead of the `QC: 2 Ruff: lint` / `QC: 2 Ruff: fix` tasks.

### PowerShell: Formatter (Invoke-Formatter) + Linting (PSScriptAnalyzer)

- **Configuration**: `scripts/powershell/PoshQC/settings/pssa.settings.psd1`
- **Install tools (once)**: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/install-powershell-tools.ps1`
- **Format**: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/format-powershell.ps1`
- **Lint**: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1`
- Prefer these commands instead of the `QC: PowerShell: format` / `QC: PowerShell: analyze` tasks.

### Type Checking: Pyright

- **Configuration**: `pyrightconfig.json`
- **Type checking mode**: Strict
- **Target**: All custom code (src, tests, scripts)

**Run type checker:**

```bash
poetry run pyright
```
Prefer the command above instead of the `QC: 3 Pyright: type-check` task.

### Testing: Pytest

- **Configuration**: `pyproject.toml` `[tool.pytest.ini_options]`
- **Coverage**: pytest-cov plugin

**Run tests:**

```bash
poetry run pytest
```

**Run with coverage:**

```bash
poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing
```
Prefer these commands instead of the `QC: 4 Pytest: run tests` / `QC: 4 Pytest: run tests with coverage` tasks.

### JSON Config Quality: jq + jsonschema

- **Governed globs**: `.vscode/*.json`, `.vscode/**/*.json`, `.devcontainer/*.json`, `scripts/**/*.json`, `docs/**/*.json`, `examples/**/*.json`
- **Excludes**: `data/**`, `artifacts/**`, `htmlcov/**`, `coverage*/**`, `**/node_modules/**`
- **Formatter**: strict JSON only (no comments); sorts keys deterministically via jq.

  ```bash
  poetry run python -m scripts.dev_tools.format_json          # rewrite in place
  poetry run python -m scripts.dev_tools.format_json --check  # verify only
  ```

- **Validator**: requires `$schema` on governed files; caches schemas under `.cache/schemas`.

  ```bash
  poetry run python -m scripts.dev_tools.validate_json                # validate governed files
  poetry run python -m scripts.dev_tools.validate_json --verbose      # show per-file status
  poetry run python -m scripts.dev_tools.validate_json --cache-dir .cache/schemas
  ```

- **VS Code tasks**: `JSON: format`, `JSON: validate`

### PowerShell Testing: Pester

- **Configuration**: `scripts/powershell/PoshQC/settings/pester.runsettings.psd1`
- **Run tests**: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1`
- Prefer the command above instead of the `QC: PowerShell: test (Pester)` task.

### Shell QC (Bash)

```bash
poetry run shell-qc check
poetry run shell-qc format
poetry run shell-qc test
poetry run shell-qc test --coverage
```

Requires external tools: `shfmt`, `shellcheck`, `bats-core` (optional, only for tests).

Optional coverage:

- `--coverage` runs `bats` under `kcov` and writes a Cobertura report to:
  `artifacts/pester/kcov/kcov-merged/cov.xml`
- This is intended to be auto-discovered by the VS Code Coverage Gutters extension.

Coverage requires the additional tool: `kcov`.
In the devcontainer these tools are installed by the container image (once you add them).
Outside the devcontainer, install them on your OS or use WSL on Windows for best results.

## Integrated Workflows

### Run All Checks (Sequential)

Run JSON format → JSON validate → shell-qc format/check/test → Black → Ruff → Pyright → Pytest → PowerShell format/analyze/test in sequence:

```bash
poetry run python -m scripts.dev_tools.format_json
poetry run python -m scripts.dev_tools.validate_json
poetry run python -m scripts.dev_tools.shell_qc format
poetry run python -m scripts.dev_tools.shell_qc check
poetry run python -m scripts.dev_tools.shell_qc test
poetry run black .
poetry run ruff check
poetry run pyright
poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/format-powershell.ps1
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1
```

### Fix All (Automated)

Runs JSON format → JSON validate → shell-qc format/check/test → Black formatting + Ruff auto-fixes + verification → Pyright → Pytest (with coverage) → PowerShell format/analyze/test via the Python entry point:

```bash
poetry run python -m scripts.dev_tools.fix_all

# PowerShell wrapper (delegates to Python for compatibility)
pwsh scripts/dev-tools/fix-all.ps1
```
Use the commands above instead of the `QC: 0 Fix All` task.

## Secret Management

### API Keys and Credentials

**NEVER commit secrets to the repository:**

- No API keys in code, config files, or `.env` files
- Use environment variables or secure secret storage

### OpenAI API Key Management

**Storage:** LastPass (recommended)

- Store in a secure note named: `Lexile OpenAI Key`
- Login with: `lpass login`

**Loading secrets:**

```powershell
# Load from LastPass
pwsh src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/load-openai-key.ps1 -ItemName "Lexile OpenAI Key"

# Options:
# -UsePasswordField    # Pull from password field instead of note
# -EnvVar "VAR_NAME"   # Override target environment variable
# -PrintOnly           # Output without exporting (for CI)
```

**VS Code task:** `Load OpenAI Key`

**Environment variable usage:**

```bash
# Set environment variable
export OPENAI_API_KEY="your-key-here"

# Or pass via config
--openai-api-key "your-key-here"

# Or specify env var name in config
openai:
  api_key_env: OPENAI_API_KEY
```

**For CI/CD:**

- Use platform secret managers (GitHub Actions Secrets, Azure Key Vault, etc.)
- Inject environment variables before running CLI
- Do NOT script LastPass logins in CI

### Configuration Precedence

API keys are resolved in this order:

1. Explicit CLI flag: `--openai-api-key`
2. Config file: `openai.api_key`
3. Environment variable: `os.environ[openai.api_key_env]`

## OER Curation Workflows

### CK-12 Workflow

The CK-12 catalog and enrichment pipeline derives artifact types from `Content_URL` path prefixes. Supported artifact types are:

- `cbook` (from `flexbooks.ck12.org/cbook/`)
- `book` (from `www.ck12.org/book/`)
- `tebook` (from `www.ck12.org/tebook/`)
- `workbook` (from `www.ck12.org/workbook/`)
- `quizbook` (from `www.ck12.org/quizbook/`)

These artifact types are automatically detected during catalog parsing and are critical for the Perma API to correctly retrieve revision JSON data during enrichment. The artifact type is passed to the Perma API to fetch the correct content format.

For detailed step-by-step instructions, see `docs/source-curation-guide.md`.

## Repository Navigation

Use `scripts/dev-tools/tree.ps1` to print a directory tree for quick inspection (entries marked with the Windows `Hidden` attribute are included by default; use `-IncludeHidden:$false` to suppress them).

```powershell
pwsh scripts/dev-tools/tree.ps1
```

- Show only directories:

```powershell
pwsh scripts/dev-tools/tree.ps1 -DirectoriesOnly
```

- Exclude entries or include dotfiles as needed, for example:

```powershell
pwsh scripts/dev-tools/tree.ps1 -Exclude @(".git", "node_modules", "artifacts") -IncludeHidden:$false
```

## Git Workflow

### Commit Context Collection

Collect context for commits or pull requests:

```powershell
# Collect commit context
poetry run python -m scripts.dev_tools.collect_commit_context --output artifacts/commit_context.txt

# Collect PR context
poetry run python -m scripts.dev_tools.pr_context.collector --base origin/main
```

Output saved to `artifacts/` directory.

## Atomic Execution Agent

The `atomic_executor` tool orchestrates the execution of atomic plans (`docs/features/active/*/plan.md`) using GitHub Copilot CLI. It enforces repo policies, runs QC checks, and handles retries.

### Usage

**Execute next available task:**
```bash
poetry run python -m scripts.dev_tools.atomic_executor.cli execute \
  --path docs/features/active/feature-name
```

**Execute ALL tasks in order (autonomous mode):**
```bash
poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all \
  --path docs/features/active/feature-name \
  --max-fix-attempts 3   # 0 for infinite retries
```

**Session behavior notes:**
- The executor invokes Copilot CLI with the `atomic_executor` agent profile (`--agent atomic_executor`).
- The first task starts a new session; subsequent tasks use `--continue` when supported to preserve context.
- `execute-all` acquires a single-run lock at `.agent_logs/executor.lock` to prevent concurrent runs from resuming unrelated sessions. Remove the lock file only if a prior run crashed and you have verified no other executor is active.
- **Pre-flight QC**: Before executing any tasks, the executor runs a full QC check (Black, Ruff, Pyright, Pytest). If baseline QC fails, Copilot is invoked to fix the issues and must run the toolchain itself until passing. Use `--skip-preflight-qc` to bypass this check.
- Prompt size and line count are logged; a warning is emitted when prompts exceed 15KB so you can trim context.
- **Graceful shutdown**: Pressing Ctrl+C (SIGINT) or sending SIGTERM triggers a graceful shutdown that releases the executor lock before exiting.
- Headless defaults allow shell commands and all paths, and the workspace is added to Copilot CLI trusted_folders. Use `--no-copilot-allow-shell`, `--no-copilot-allow-all-paths`, `--copilot-allow-all-urls`, or `--no-copilot-trust-workspace` to override.

**Resume execution (next unchecked task):**
```bash
poetry run python -m scripts.dev_tools.atomic_executor.cli resume \
  --path docs/features/active/feature-name
```

**Generate Prompt Only:**
```bash
poetry run python -m scripts.dev_tools.atomic_executor.cli execute \
  --path docs/features/active/feature-name \
  --print-prompt
```

### Copilot CLI throttling controls

The executor self-regulates GitHub Copilot CLI usage based on **call frequency** (calls per time window), not token usage.

Key flags (safe defaults):

- `--copilot-cli-max-calls-per-window` (default: 6)
- `--copilot-cli-window-seconds` (default: 60)
- `--copilot-cli-backoff-base-seconds` (default: 2)
- `--copilot-cli-backoff-max-seconds` (default: 60)
- `--copilot-cli-output-tail-bytes` (default: 4096) — bounded output capture used for throttle classification
- `--copilot-cli-max-retries` (default: 8) — bounded retries on detected throttling

Platform note:

- GitHub Copilot CLI native Windows PowerShell support is experimental; WSL is recommended on Windows when available.

### TDD Red Workflow (`[expect-fail]` tag)

For TDD-first development, the executor supports tasks that intentionally create failing tests. Annotate the task title with the `[expect-fail]` tag:

```markdown
- [ ] [P1-T1] [expect-fail] Add failing regression test for slug extraction
```

Behavior:
- The tag inverts pytest success criteria: **pytest must fail** for the task to succeed.
- Black, Ruff, and Pyright must still pass (formatting, linting, and type-checking are not inverted).
- If all QC passes (unexpected green), the executor retries (Copilot should ensure the test actually fails).
- On success, the executor logs: `Task {task_id} failed as expected (TDD Red). Verified.`

This enables proper TDD flow where the plan can include a failing-test task followed by an implementation task that makes it pass.

## VS Code Integration

### Recommended Extensions

- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter (ms-python.black-formatter)
- Ruff (charliermarsh.ruff)

### Tasks

All development tasks are available via:

- Command Palette: `Tasks: Run Task`
- Keyboard: `Ctrl+Shift+B` (default build task)

**Available tasks:**

- `Black: format`
- `Ruff: lint`
- `Ruff: fix`
- `Pyright: type-check`
- `Pytest: run tests` (default)
- `Pytest: run tests with coverage`
- `Run All Checks` (sequential)
- `Fix All` (Black + Ruff auto-fix)
- `Load OpenAI Key`

### Debug Configurations

See `.vscode/launch.json` for debug configurations:

- CLI commands (analyze, rewrite, corpus, calibration, etc.)
- Production scripts (Gutenberg query builder, etc.)

## Troubleshooting

### Poetry Lock Issues

```bash
poetry lock --no-update
```

### Import Errors

Ensure package is installed in development mode:

```bash
poetry install
```

### Type Checking Errors

Run Pyright with verbose output:

```bash
poetry run pyright --verbose
```

### Test Failures

Run specific test with verbose output:

```bash
poetry run pytest -vv tests/test_specific.py::test_function
```

## Documentation Standards

For comprehensive coding standards, testing policies, and development workflow, see:

- **Coding standards (general)**: [`.github/instructions/general-code-change.instructions.md`](../.github/instructions/general-code-change.instructions.md)
- **Coding standards (Python)**: [`.github/instructions/python-code-change.instructions.md`](../.github/instructions/python-code-change.instructions.md)
- **Testing policy (general)**: [`.github/instructions/general-unit-test.instructions.md`](../.github/instructions/general-unit-test.instructions.md)
- **Testing policy (Python)**: [`.github/instructions/python-unit-test.instructions.md`](../.github/instructions/python-unit-test.instructions.md)
- **CI/CD**: [`docs/ci-documentation.md`](ci-documentation.md)
