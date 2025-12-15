# Docker Dev Container Setup

This directory contains the Docker Dev Container configuration for the Lexile Corpus Tuner project.

## Prerequisites

- **Docker Desktop** installed and running
- **VS Code** with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed

## Quick Start

1. Open this workspace in VS Code
2. Press `F1` and select **Dev Containers: Reopen in Container**
3. Wait for the container to build and configure (first time ~5-10 minutes)
4. Once complete, you'll have a fully configured development environment!

## What's Included

### Base Environment
- **Debian Bookworm** (latest stable)
- **Python 3.13** with Poetry package manager
- **PowerShell 7.5+** for cross-platform scripting
- **Git** and **GitHub CLI** (gh)
- **cloc** for code metrics
- **actionlint** for GitHub Actions workflow linting

### Python Tooling
- **Black** (code formatter)
- **Ruff** (fast linter)
- **Pyright** (type checker)
- **Pytest** (test runner with coverage)
- All project dependencies from `pyproject.toml`

### PowerShell Tooling
- **PSScriptAnalyzer** (linter)
- **Pester** (test framework)
- **PoshQC** module (from workspace)

### VS Code Extensions
- Python, Pylance, Black Formatter, Ruff
- PowerShell
- GitLens, GitHub Pull Requests
- EditorConfig, Code Spell Checker

## Container Features

### Virtual Environment
Poetry creates a virtual environment at `/workspace/.venv` which is:
- Mounted from your local `.venv` folder for persistence
- Automatically activated in the terminal
- Configured as the default Python interpreter

### Terminal Defaults
- Default shell: **PowerShell** (`pwsh`)
- Bash available as alternative

### All Repository Tasks Available
All VS Code tasks from `.vscode/tasks.json` work in the container:
- `QC: 1 Black: format`
- `QC: 2 Ruff: lint`
- `QC: 3 Pyright: type-check`
- `QC: 4 Pytest: run tests`
- `PoshQC: 1 format`
- `PoshQC: 2 analyze`
- `PoshQC: 4 test (Pester)`
- `QC: 5 Run All Checks`

## Rebuilding the Container

If you modify the Dockerfile or devcontainer.json:

1. Press `F1` and select **Dev Containers: Rebuild Container**
2. Or, from outside the container: **Dev Containers: Rebuild and Reopen in Container**

## Customization

### Adding Extensions
Edit `.devcontainer/devcontainer.json` → `customizations.vscode.extensions[]`

### Changing Python/Tool Versions
Edit `.devcontainer/Dockerfile` → `FROM` line and install commands

### Post-Create Steps
Edit `.devcontainer/post-create.sh` to add custom setup commands

## Secrets and API Keys

The container does **not** include secrets. To use OpenAI API:

1. Inside the container, install LastPass CLI or use environment variables
2. Run the load script: `pwsh scripts/dev-tools/load-openai-key.ps1`
3. Or set manually: `export OPENAI_API_KEY="your-key"`

## Troubleshooting

### Container won't start
- Ensure Docker Desktop is running
- Check Docker Desktop resources (RAM: 4GB+, CPU: 2+ cores)
- View logs: `F1` → **Dev Containers: Show Container Log**

### Poetry install fails
- Rebuild container: `F1` → **Dev Containers: Rebuild Container**
- Check `poetry.lock` is committed to repo

### Extensions not loading
- Reopen window: `F1` → **Dev Containers: Reopen in Container**
- Check extension compatibility with container architecture

### Performance issues on Windows
- Ensure workspace is on a local drive (not network share)
- Consider WSL 2 backend for Docker Desktop
- The `.venv` mount helps by keeping Python packages on host filesystem

## Benefits Over Local Development

✅ **Consistent environment** across team members  
✅ **No pollution** of host system with dev tools  
✅ **Isolated Python/PowerShell** versions and packages  
✅ **Easy onboarding** - one command to get started  
✅ **Reproducible builds** - same environment every time  
✅ **Cross-platform** - works on Windows, macOS, Linux  

## Additional Resources

- [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [Dev Container Specification](https://containers.dev/)
- [Docker Desktop Documentation](https://docs.docker.com/desktop/)
