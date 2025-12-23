# Docker Dev Container - Quick Reference

## Starting the Dev Container

### First Time Setup
1. Ensure Docker Desktop is running
2. In VS Code, press `F1` (or `Ctrl+Shift+P`)
3. Type and select: **Dev Containers: Reopen in Container**
4. Wait for the build process (~5-10 minutes)
5. Container will automatically run post-create setup

### Subsequent Uses
- Just open the workspace - VS Code will prompt to reopen in container
- Or use `F1` → **Dev Containers: Reopen in Container**

## Common Commands

### In Terminal (inside container)
```bash
# Run all QC checks
pwsh scripts/dev-tools/fix-all.ps1

# Python formatting and linting
poetry run black .
poetry run ruff check --fix

# Type checking and testing
poetry run pyright
poetry run pytest

# PowerShell QC
pwsh -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCFormat -Root ."
pwsh -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCAnalyze -Root ."
pwsh -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."

# Run CLI tools
poetry run lexile-tuner --help
poetry run lexile-scoring-model-pipeline --help
```

### VS Code Tasks
All existing tasks work in the container. Access via:
- `Ctrl+Shift+B` (default build task)
- `F1` → **Tasks: Run Task**

## Managing the Container

### Rebuild Container
When you change Dockerfile or devcontainer.json:
```
F1 → Dev Containers: Rebuild Container
```

### Stop Container
```
F1 → Dev Containers: Close Remote Connection
```

### View Container Logs
```
F1 → Dev Containers: Show Container Log
```

### Open New Terminal
```
Ctrl+` or Terminal → New Terminal
```

## File Persistence

### What Persists
- Your workspace files live on a named volume (`${localWorkspaceFolderBasename}-workspace` at `/workspaces/lexile-corpus-tuner`)
- A read-only host bind (`/workspaces/lexile-corpus-host`) is used only to seed the volume on first create; `.venv`, `artifacts`, and `data` are excluded from the seed to avoid long copies. Rebuild the container if you need to refresh from host.
- Dedicated artifacts folder bound to host sibling path `${localWorkspaceFolder}/../lexile-artifacts` at `/workspaces/lexile-artifacts`
- `.venv` folder (mounted from host)
- Git configuration
- Installed VS Code extensions

### What Doesn't Persist
- Global system packages (outside workspace)
- Shell history (by default)
- Temporary container files

## Troubleshooting

### Container Build Fails
1. Check Docker Desktop is running and has resources (4GB+ RAM)
2. View error logs: `F1` → **Dev Containers: Show Container Log**
3. Try: `F1` → **Dev Containers: Rebuild Container Without Cache**

### Poetry/Python Issues
```bash
# Reinstall dependencies
poetry install --no-cache

# Clear Poetry cache
poetry cache clear pypi --all
```

### PowerShell Module Issues
```bash
# Reimport PoshQC
pwsh -Command "Import-Module -Force ./scripts/powershell/PoshQC"

# Verify modules installed
pwsh -Command "Get-Module -ListAvailable PSScriptAnalyzer, Pester"
```

### Permission Issues
The container runs as user `vscode` (non-root). If you need root:
```bash
sudo apt-get update
```

### Performance on Windows
- Workspace now runs from a named volume to avoid host bind latency; host bind is read-only and only for bootstrap
- Use WSL 2 backend in Docker Desktop settings
- Keep large artifacts in `/workspaces/lexile-artifacts` (host sibling) to avoid scanning overhead inside the code workspace
- Aggressive workspace scanners are not allowed (Task Explorer removed)
- The `.venv` mount helps performance by keeping packages on host

## Tips

### Multiple Terminals
- Default: PowerShell (`pwsh`)
- Switch to Bash: Click terminal dropdown → Select Default Profile → Bash

### Secrets Management
```bash
# Set environment variable for OpenAI
export OPENAI_API_KEY="your-key-here"

# Or use the load script
pwsh scripts/dev-tools/load-openai-key.ps1
```

### Extension Management
- Extensions auto-install from devcontainer.json (Task Explorer excluded by policy)
- Install more: Extensions panel → Install in Dev Container (avoid workspace-wide scanners)

### Customize Shell Prompt
Add to `~/.bashrc` or `~/.config/powershell/profile.ps1` inside container
(these will be lost when container is rebuilt - use post-create.sh for persistence)

## Advantages

✅ No need to install Python, Poetry, PowerShell, or tools locally  
✅ Same environment for all developers  
✅ Clean separation from host system  
✅ Easy to reset (rebuild container)  
✅ Works on Windows, macOS, Linux  
✅ All VS Code tasks and debugging work normally  

## Going Back to Local Development

```
F1 → Dev Containers: Reopen Folder Locally
```

Your local environment is unchanged - the container doesn't affect your host machine.
