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

### VS Code Extensions (allow list)
- Python, Pylance, Black Formatter, Ruff, Poetry helper, Parquet Explorer, Live Server
- PowerShell
- GitLens, GitHub Pull Requests, Git Graph, Graphite GTI
- EditorConfig, REST Client
- Coverage Gutters, Koverage, Pester Test Explorer
- Docker tools, Markdown Preview GitHub Styles, Mermaid, Rainbow CSV
- ChatGPT, GitHub Actions

### Mounts and Storage Layout
- Workspace stored on a named volume: `${localWorkspaceFolderBasename}-workspace -> /workspaces/lexile-corpus-tuner`
- Dedicated background worktree volume: `${localWorkspaceFolderBasename}-workspace-bg -> /workspaces/lexile-corpus-tuner-bg` (used by background tasks)
- Host workspace mounted read-only at `/workspaces/lexile-corpus-host` for initial bootstrap copy into the volume on first create (excludes `.venv`, `artifacts`, `data` to avoid slow transfers)
- Dedicated artifact bind mount: `${localWorkspaceFolder}/../lexile-artifacts -> /workspaces/lexile-artifacts` (keeps large data outside the scanned code workspace)
- Docker socket bind: `/var/run/docker.sock -> /var/run/docker.sock`

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
- The workspace now lives on a named volume to avoid host bind latency; a read-only host bind is only used to seed the volume on first create
- Ensure Docker Desktop uses the WSL 2 backend; keep host-side edits in sync by rebuilding the container if you need a fresh copy from the host
- Keep large artifacts under `/workspaces/lexile-artifacts` instead of inside the code volume
- Avoid aggressive workspace scanners (Task Explorer removed from allow list)

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
