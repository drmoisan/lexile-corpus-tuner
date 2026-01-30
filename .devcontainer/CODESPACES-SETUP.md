# GitHub Codespaces Setup Guide

This repository uses separate dev container configurations for **Codespaces** and **local Docker**. You must manually select the Codespaces configuration when creating your Codespace.

## Why Manual Selection?

To prevent VS Code from defaulting to the Codespaces config when opening locally, we keep both configs as non-standard filenames:
- `.devcontainer/codespaces/devcontainer.json` - For Codespaces (manual selection)
- `.devcontainer/local/devcontainer.json` - For local Docker (manual selection)

This avoids conflicts where one environment's config interferes with the other.

## Creating a Codespace with the Correct Configuration

### Method 1: GitHub Web UI (Recommended)

1. **Go to the repository on GitHub.com**
   - Navigate to: `https://github.com/drmoisan/lexile-corpus-tuner`

2. **Click the "Code" button** (green button, top right)

3. **Switch to the "Codespaces" tab**

4. **Click the "..." (three dots) or dropdown arrow** next to "Create codespace on [branch]"
   
   You may see one of these options:
   - **"Configure and create codespace"** ← Click this
   - **"New with options..."** ← Or this
   - **A gear icon** next to the create button ← Or click this

5. **In the configuration panel:**
   - Look for **"Dev container configuration"** section
   - Click the dropdown to expand options
   - **Select:** `.devcontainer/codespaces/devcontainer.json`
   - Leave other settings as default (Machine type: 2-core is fine, can upgrade later)

6. **Click "Create codespace"**

7. **Wait for build** (~5-10 minutes first time, faster after)

8. **Verify setup:**
   ```bash
   bash .devcontainer/verify-container.sh
   ```
   You should see all ✅ checks.

**Screenshot guide:** Look for these UI elements:
```
┌─────────────────────────────────────┐
│ Code ▼ (green button)               │
├─────────────────────────────────────┤
│ ┌ Local   Codespaces   ... ┐       │
│ │                            │       │
│ │ Create codespace on main  │       │
│ │    ... or ⚙              │← Click │
│ └────────────────────────────┘       │
│                                     │
│ Configuration panel appears:        │
│ ┌───────────────────────────────┐   │
│ │ Dev container configuration   │   │
│ │ ▼ .devcontainer/codespaces/   │← Select this │
│ │   devcontainer.json           │   │
│ └───────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Method 2: GitHub CLI

```bash
# Create Codespace with specific config
gh codespace create \
  --repo drmoisan/lexile-corpus-tuner \
  --branch your-branch-name \
  --devcontainer-path .devcontainer/codespaces/devcontainer.json

# List your Codespaces
gh codespace list

# Connect to the Codespace in VS Code
gh codespace code --codespace <codespace-name>
```

### Method 3: VS Code Command Palette

1. Open VS Code
2. Press `F1` or `Ctrl+Shift+P`
3. Type: **Codespaces: Create New Codespace**
4. Select repository: `drmoisan/lexile-corpus-tuner`
5. Select branch
6. **Important:** When prompted for dev container config:
   - Select `.devcontainer/codespaces/devcontainer.json`
7. Wait for build

## Expected Environment After Setup

Once your Codespace is created, you should have:

### ✅ Correct Configuration
- **OS:** Debian Bookworm (not Ubuntu)
- **Python:** 3.13.x (not 3.12.x)
- **Poetry:** 2.2.1
- **PowerShell:** 7.5+
- **Shell tools:** shellcheck, shfmt, bashdb, actionlint

### Verification Command
```bash
bash .devcontainer/verify-container.sh
```

**Expected output:**
```
✅ All checks passed!

Your dev container is correctly configured.
Using: .devcontainer/codespaces/devcontainer.json (Codespaces)
```

## Common Issues

### ❌ Issue: Codespace has wrong environment (Ubuntu, Python 3.12, no PowerShell)

**Cause:** GitHub used the default universal image instead of your custom config.

**Solution:**
1. Delete the incorrect Codespace
2. Create a new one following **Method 1** above
3. Ensure you select `.devcontainer/codespaces/devcontainer.json`

### ❌ Issue: Can't find config selection in GitHub UI

**GitHub UI has changed:**
- Look for **"Configure and create codespace"** option
- Or **"New with options"** button
- Or click the **"..." / three dots** next to "Create codespace"

**Alternative:**
- Use **Method 2 (GitHub CLI)** for explicit config path

### ❌ Issue: Build fails with Dockerfile errors

**Check:**
1. Ensure `.devcontainer/codespaces/Dockerfile` exists and is not disabled
2. Check Codespace logs for specific error
3. See `.devcontainer/TROUBLESHOOTING.md`

## VS Code Settings for Codespaces

Once in your Codespace, VS Code settings are pre-configured:
- Python formatting with Black (auto-save)
- Ruff linting
- Pyright type checking (strict mode)
- PowerShell as default terminal
- All necessary extensions pre-installed

## Switching Branches in Codespace

You can work on different branches in the same Codespace:
```bash
git fetch
git checkout your-feature-branch
```

The dev container config stays the same unless you rebuild.

## Rebuilding Your Codespace

If you update the dev container configuration:
```
F1 → Codespaces: Rebuild Container
```

Or via Codespaces menu (click your Codespace name in status bar).

## Codespace vs Local Development

| Aspect | Codespaces | Local Docker |
|--------|------------|--------------|
| **Config** | `.devcontainer/codespaces/` | `.devcontainer/local/` |
| **Selection** | Manual via GitHub UI | Manual in VS Code |
| **Environment** | Cloud VM (2-4 cores, 8GB) | Your Docker resources |
| **Performance** | Network-dependent | Local disk speed |
| **Setup Time** | ~5-10 min first time | ~5-10 min first time |
| **Cost** | GitHub free tier limit | Free (uses your hardware) |

Both environments provide identical tooling and configurations.

## Quick Reference

### Create Codespace (correct config)
```bash
gh codespace create --repo drmoisan/lexile-corpus-tuner \
  --devcontainer-path .devcontainer/codespaces/devcontainer.json
```

### Verify Environment
```bash
bash .devcontainer/verify-container.sh
```

### View Config Path (inside Codespace)
```bash
cat /workspaces/.codespaces/shared/devcontainer.json
```

### Delete Codespace
```bash
gh codespace delete --codespace <name>
```

Or via GitHub.com → Code → Codespaces → ... → Delete

## Additional Resources

- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces)
- [Dev Container Specification](https://containers.dev/)
- Local setup: `.devcontainer/README.md`
- Config comparison: `.devcontainer/CONFIG-GUIDE.md`
