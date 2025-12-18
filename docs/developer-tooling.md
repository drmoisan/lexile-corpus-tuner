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

**VS Code task:** `Black: format`

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

**VS Code tasks:**

- `Ruff: lint` - Check for issues
- `Ruff: fix` - Auto-fix issues

### PowerShell: Formatter (Invoke-Formatter) + Linting (PSScriptAnalyzer)

- **Configuration**: `scripts/powershell/PoshQC/settings/pssa.settings.psd1`
- **Install tools (once)**: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/install-powershell-tools.ps1`
- **Format**: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/format-powershell.ps1`
- **Lint**: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1`
- **VS Code tasks**: `QC: PowerShell: format`, `QC: PowerShell: analyze`

### Type Checking: Pyright

- **Configuration**: `pyrightconfig.json`
- **Type checking mode**: Strict
- **Target**: All custom code (src, tests, scripts)

**Run type checker:**

```bash
poetry run pyright
```

**VS Code task:** `Pyright: type-check`

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

**VS Code tasks:**

- `Pytest: run tests` - Run all tests (default test task)
- `Pytest: run tests with coverage` - Run with coverage report

### PowerShell Testing: Pester

- **Configuration**: `scripts/powershell/PoshQC/settings/pester.runsettings.psd1`
- **Run tests**: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1`
- **VS Code task**: `QC: PowerShell: test (Pester)`

## Integrated Workflows

### Run All Checks (Sequential)

Runs Black → Ruff → Pyright → Pytest in sequence:

```bash
# VS Code task
Run All Checks
```

### Fix All (Automated)

Runs Black formatting + Ruff auto-fixes + verification + Pyright + Pytest (with coverage) via the Python entry point:

```bash
poetry run python -m scripts.dev_tools.fix_all

# PowerShell wrapper (delegates to Python for compatibility)
pwsh scripts/dev-tools/fix-all.ps1
```

**VS Code task:** `Fix All`

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

## Code Metrics

### Lines of Code

Count source lines using cloc:

```bash
# PowerShell
pwsh scripts/dev-tools/run-cloc.ps1

# Bash
bash scripts/dev-tools/run-cloc.sh
```

**VS Code tasks:**

- `Run cloc (PowerShell)`
- `Run cloc (Bash)`

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
pwsh scripts/dev-tools/collect-commit-context.ps1

# Collect PR context
pwsh scripts/dev-tools/collect-pull-request-context.ps1
```

Output saved to `artifacts/` directory.

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
- `Run cloc (PowerShell)`
- `Run cloc (Bash)`

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
