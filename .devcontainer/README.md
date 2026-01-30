# Docker Dev Container Setup

This directory contains the Docker Dev Container configuration for the Lexile Corpus Tuner project.

> **Important**: This repo uses **separate configurations** for GitHub Codespaces and local Docker. Both require **manual selection** to avoid conflicts.

## Prerequisites

- **Docker Desktop** installed and running (for local development)
- **VS Code** with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed (for local development)
- **OR** use **GitHub Codespaces** (no local setup needed)

## Quick Start

### GitHub Codespaces

**⚠️ You must manually select the Codespaces configuration:**

1. Go to repository on GitHub
2. Click **Code** → **Codespaces** → **... (three dots)**
3. Select **"Configure and create codespace"**
4. Choose dev container: **`.devcontainer/codespaces/devcontainer.json`**
5. Click **"Create codespace"**
6. Wait for build (~5-10 minutes first time)
7. Verify: `bash .devcontainer/verify-container.sh`

**📖 Detailed instructions:** See [CODESPACES-SETUP.md](CODESPACES-SETUP.md)

### Local Docker Development

1. Ensure Docker Desktop is running
2. Open this workspace in VS Code
3. Press `F1` → **Dev Containers: Open Folder in Container**
4. Navigate to and select: **`.devcontainer/local/devcontainer.json`**
5. Wait for build (~5-10 minutes first time)
6. Verify: `bash .devcontainer/verify-container.sh`

**Why manual selection?** To prevent one environment's config from interfering with the other, we keep both as non-standard filenames that require explicit selection.

## What's Included

### Base Environment
- **Debian Bookworm** (latest stable)
- **Python 3.13** with Poetry package manager
- **PowerShell 7.5+** for cross-platform scripting
- **Git** and **GitHub CLI** (gh)
- **Graphite CLI** (`gt`) for the Graphite VS Code extension
- **GitHub Copilot CLI** (`copilot`) for the `atomic_executor` tool
- **actionlint** for GitHub Actions workflow linting
- **bashdb** for Bash debugging

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

### Common Commands (preferred over VS Code tasks)
Use the Poetry and PowerShell commands directly instead of the `QC: ...` tasks:
- `poetry run black .`
- `poetry run ruff check`
- `poetry run pyright`
- `poetry run pytest`
- `poetry run python -m scripts.dev_tools.fix_all`
- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/format-powershell.ps1`
- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1`
- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1`

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
