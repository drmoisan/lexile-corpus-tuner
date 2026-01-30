# Dev Container Configuration Guide

## Overview

This repository uses **separate dev container configurations** for Codespaces and local development. Both require **manual selection** to prevent conflicts.

**Why manual selection?** Having a config at `.devcontainer/devcontainer.json` causes VS Code to use it by default for local dev, even when you want the local-optimized version. By keeping both as non-standard filenames, you explicitly choose which to use.

## Configurations

1. **Codespaces** (`.devcontainer/codespaces/devcontainer.json`) - Manual selection required
2. **Local Docker** (`.devcontainer/local/devcontainer.json`) - Manual selection required

Both provide identical tooling (Debian Bookworm, Python 3.13, PowerShell 7.5+, etc.)

## Quick Start

### For GitHub Codespaces

**You must manually select the config:**

1. GitHub.com → Repository → **Code** → **Codespaces**
2. Click **... (three dots)** → **"Configure and create codespace"**
3. Select: `.devcontainer/codespaces/devcontainer.json`
4. Click "Create codespace"

📖 **Detailed guide:** [CODESPACES-SETUP.md](CODESPACES-SETUP.md)

### For Local VS Code

1. Docker Desktop running
2. VS Code → `F1` → **Dev Containers: Open Folder in Container**
3. Navigate to: `.devcontainer/local/devcontainer.json`
4. Select it and wait for build

## Environment Details

Both configurations provide:

### Base Environment
- Debian Bookworm (latest stable)
- Python 3.13 with Poetry 2.2.1
- PowerShell 7.5+ with PSScriptAnalyzer and Pester
- Git, GitHub CLI, actionlint, bashdb

### Python Tooling
- Black, Ruff, Pyright, Pytest
- All dependencies from pyproject.toml

### VS Code Extensions
- Python, Pylance, Black Formatter, Ruff
- PowerShell, GitLens, GitHub Pull Requests
- Coverage tools, Docker tools, and more

## Verifying Your Configuration

After container starts, check which config was used:

```bash
# Check environment
cat /etc/os-release  # Should show Debian Bookworm
python --version     # Should show 3.13.x
pwsh --version       # Should show PowerShell 7.5+
poetry --version     # Should show 2.2.1

# Check if running in Codespaces
echo $CODESPACES     # "true" if in Codespaces, empty if local
```

## Rebuilding

If you modify configuration files:

```
F1 → Dev Containers: Rebuild Container
```

For Codespaces:
```
Codespace menu → Rebuild Container
```

## Files Reference

| File | Purpose | Used By |
|------|---------|---------|
| `.devcontainer/devcontainer.json` | Codespaces configuration | GitHub Codespaces |
| `.devcontainer/codespaces/devcontainer.json` | Original Codespaces config (reference) | (archived) |
| `.devcontainer/codespaces/Dockerfile` | Codespaces Dockerfile | Both configs |
| `.devcontainer/local/devcontainer.json` | Local Docker configuration | VS Code local |
| `.devcontainer/local/Dockerfile` | Local Dockerfile | Local config |
| `.devcontainer/post-create.sh` | Post-create setup script | Both configs |
| `.devcontainer/devcontainer.json.disabled` | Previous single config (disabled) | (archived) |

## Troubleshooting

### "Wrong configuration was used"

**Symptom**: Container has wrong Python version, missing PowerShell, etc.

**For Codespaces**:
- Delete and recreate the Codespace
- The `.devcontainer/devcontainer.json` should be used automatically

**For Local**:
- Close container: `F1` → **Dev Containers: Close Remote Connection**
- Explicitly open: `F1` → **Dev Containers: Open Folder in Container**
- Navigate to and select: `.devcontainer/local/devcontainer.json`

### "VS Code uses wrong config by default"

VS Code prioritizes configs in this order:
1. `.devcontainer/devcontainer.json` (if exists)
2. `.devcontainer.json` (root)
3. Prompts for selection if multiple found

To force selection:
```
F1 → Dev Containers: Open Folder in Container
```

## Additional Resources

- Main README: `.devcontainer/README.md`
- Quick Reference: `.devcontainer/QUICKSTART.md`
- Troubleshooting: `.devcontainer/TROUBLESHOOTING.md`
