# Continuous Integration Documentation

## Overview

This project uses GitHub Actions for continuous integration to enforce code quality standards, run tests, and validate builds across multiple Python versions.

## CI Pipeline

### Jobs

#### 1. Code Quality & Tests (`quality-checks7`)

Runs on: Python 3.10, 3.11, 3.12, 3.13

**Steps:**
- **Formatting**: Black checks code formatting (line length: 88)
- **Linting**: Ruff enforces code quality rules
- **Type Checking**: Pyright validates type annotations (all code must be fully typed)
- **Testing**: Pytest runs all tests with coverage reporting
- **Lock File Validation**: Ensures `poetry.lock` is in sync with `pyproject.toml`

**Requirements:**
- All checks must pass (continue-on-error: false)
- Code coverage is uploaded to Codecov for Python 3.13
- Poetry dependencies are cached for faster builds

#### 2. Security Scanning (`security-scan`)

Runs on: Python 3.13

**Steps:**
- Uses `safety` to check for known security vulnerabilities in dependencies
- Runs as informational (continue-on-error: true) but failures should be investigated

#### 3. Documentation Validation (`docs-validation`)

**Steps:**
- Validates README.md exists and is not empty
- Checks for LICENSE file
- Verifies instruction documents are present

#### 4. Build Check (`build-check`)

**Steps:**
- Builds the package using Poetry
- Installs the built wheel in a clean venv
- Verifies both CLI entry points work (`lexile-tuner`, `text-difficulty-pipeline`)

## Local Development

### Running All Checks Locally

Use the provided task:
```powershell
# Run all checks sequentially
pwsh ./scripts/dev-tools/fix-all.ps1

# Or use VS Code task
# Ctrl+Shift+P -> "Tasks: Run Task" -> "Run All Checks"
```

### Individual Tools

```powershell
# Format code
poetry run black .

# Lint code
poetry run ruff check

# Auto-fix lint issues
poetry run ruff check --fix

# Type check
poetry run pyright

# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=src/lexile_corpus_tuner --cov-report=term-missing
```

### Pre-commit Hooks

Install pre-commit hooks to catch issues before committing. Use the Poetry environment so the hook versions match CI:

```powershell
# Install pre-commit into the Poetry environment (once per machine)
poetry run pip install pre-commit

# Install git hooks
poetry run pre-commit install

# Run manually on all files
poetry run pre-commit run --all-files
```

The hooks will:
- Format code with Black on commit
- Lint with Ruff (with auto-fix) on commit
- Type check with Pyright on commit
- Run fast tests on push

## Policy Enforcement

### Coding Standards (from code-change.instructions.md)

All code must:
1. **Be formatted with Black** (default settings, 88 char line length)
2. **Pass Ruff linting** (no rule disabling except with justification)
3. **Be fully type-annotated** and pass Pyright (no `Any` unless documented)
4. **Have test coverage** for all new logic

### Design Principles

1. **Simplicity first** - Prefer clear, readable code over cleverness
2. **Reusability** - Factor out common logic
3. **Extensibility** - Design for extension without breaking changes
4. **Separation of concerns** - Keep pure logic separate from I/O

### Classes vs Functions

- **Use classes for**: Domain concepts, stateful workflows, interfaces with multiple implementations
- **Use functions for**: Pure, stateless helpers and transformations
- **Prefer `@dataclass`** for value objects

### Error Handling

- Fail fast with specific exceptions
- No silent error suppression
- Use logging instead of `print()`
- Enforce invariants at construction time

### Module Structure

- Keep modules cohesive (single purpose)
- Small public API surface
- Use `_` prefix for internal helpers
- Absolute imports within project

## Secrets Management

**Never commit secrets to the repository.**

For local development with OpenAI:
```powershell
# Store in LastPass, then load on demand
pwsh ./scripts/dev-tools/load-openai-key.ps1 -ItemName "Lexile OpenAI Key"
```

For CI:
- Store secrets in GitHub Actions Secrets
- Reference as `${{ secrets.SECRET_NAME }}`
- Never log or expose secrets in CI output

## Troubleshooting

### CI Failing on Formatting

```powershell
# Fix locally
poetry run black .
git add -u
git commit -m "Apply Black formatting"
```

### CI Failing on Linting

```powershell
# Auto-fix issues
poetry run ruff check --fix
git add -u
git commit -m "Fix linting issues"
```

### CI Failing on Type Checking

```powershell
# Check locally first
poetry run pyright

# Fix type issues, then commit
git add -u
git commit -m "Fix type annotations"
```

### poetry.lock Out of Sync

```powershell
# Update lock file without changing versions
poetry lock --no-update

# Or update to latest compatible versions
poetry update

git add poetry.lock
git commit -m "Update poetry.lock"
```

### Tests Failing

```powershell
# Run tests with verbose output
poetry run pytest -v

# Run specific test
poetry run pytest tests/test_specific.py -v

# Run with debugger on failure
poetry run pytest --pdb
```

## Branch Protection Recommendations

For production branches (`main`, `develop`), enable:
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Require linear history
- Required checks:
  - Code Quality & Tests matrix (`quality-checks7`)
  - security-scan
  - docs-validation
  - build-check

## Performance

- Poetry dependencies are cached using `actions/cache@v4`
- Cache key includes Python version and `poetry.lock` hash
- Typical cold build: ~3-5 minutes
- Typical cached build: ~1-2 minutes

## Matrix Testing

Tests run on Python 3.10, 3.11, 3.12, and 3.13 to ensure compatibility.

If you need to add or drop a Python version, update:
1. `.github/workflows/ci.yml` (matrix strategy)
2. `pyproject.toml` (python version constraint)
3. `.python-version` (if present)
