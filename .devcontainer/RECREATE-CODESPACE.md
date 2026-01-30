# Recreating Your Codespace with Correct Configuration

Your current Codespace is using GitHub's default universal image. Here's how to fix it with manual config selection.

## Current State (Verified)
- ⚠️ OS: Ubuntu 24.04 (expected: Debian Bookworm)
- ⚠️ Python: 3.12.1 (expected: 3.13.x)
- ❌ PowerShell: Not installed
- ❌ Shell tools: Missing shellcheck, shfmt, bashdb, actionlint

## Solution: Create New Codespace with Manual Config Selection

### Why Manual Selection?

To prevent VS Code from defaulting to the Codespaces config when opening locally, we keep configs as non-standard filenames that require explicit selection:
- `.devcontainer/codespaces/devcontainer.json` (not at standard location)
- `.devcontainer/local/devcontainer.json` (for local development)

This means **you must manually select** the Codespaces config when creating.

### Step-by-Step Instructions

1. **Delete current Codespace**
   - GitHub.com → Code → Codespaces
   - Find your Codespace → **...** → **Delete**

2. **Create new Codespace with manual selection:**
   - GitHub.com → Repository → **Code** → **Codespaces** tab
   - Click **... (three dots)** next to "Create codespace"
   - Select **"Configure and create codespace"**
   
3. **Select configuration:**
   - In the panel, find **"Dev container configuration"**
   - Click to expand the dropdown
   - Select: **`.devcontainer/codespaces/devcontainer.json`**
   
4. **Create:**
   - Leave other settings as default
   - Click **"Create codespace"**
   - Wait ~5-10 minutes for build

5. **Verify:**
   ```bash
   bash .devcontainer/verify-container.sh
   ```

### Alternative: GitHub CLI Method

```bash
# Delete old Codespace
gh codespace list
gh codespace delete --codespace <name-of-wrong-codespace>

# Create new with explicit config path
gh codespace create \
  --repo drmoisan/lexile-corpus-tuner \
  --branch feature/populate-open-stax-ck-12-manifest-#73 \
  --devcontainer-path .devcontainer/codespaces/devcontainer.json
```

## After Rebuild Checklist

Run these commands to verify everything works:

```bash
# 1. Verify environment
bash .devcontainer/verify-container.sh

# 2. Check Python tools
python --version         # Should be 3.13.x
poetry --version         # Should be 2.2.x
pwsh --version          # Should be PowerShell 7.x

# 3. Verify project setup
poetry install          # Install dependencies
poetry run pytest       # Run tests

# 4. Verify PowerShell
pwsh -Command "Get-Module -ListAvailable PSScriptAnalyzer, Pester"

# 5. Test QC tools
poetry run black --version
poetry run ruff --version
poetry run pyright --version
```

## Expected Output After Rebuild

The verification script should show:
```
✅ All checks passed!

Your dev container is correctly configured.
Using: .devcontainer/devcontainer.json (Codespaces)
```

And you should have:
- Debian Bookworm OS
- Python 3.13.x
- PowerShell 7.5+
- Poetry 2.2.1
- All shell tools (shellcheck, shfmt, bashdb, actionlint)

## Troubleshooting

### Rebuild doesn't fix it
- Try **Option 2** (delete and recreate)
- Ensure you're on a branch with the latest `.devcontainer/devcontainer.json`

### Build fails
- Check Codespace logs in the terminal
- See `.devcontainer/TROUBLESHOOTING.md`
- Report issue with build logs

### Still shows wrong config
- Verify `.devcontainer/devcontainer.json` exists and is not disabled
- Check file points to `codespaces/Dockerfile`
- Confirm you're in a Codespace (not local Docker)

## Why This Happened

GitHub Codespaces looks for configuration in this priority order:
1. `.devcontainer/devcontainer.json` ← **NOW EXISTS** ✅
2. `.devcontainer.json` (root)
3. Falls back to universal image ← **WAS HAPPENING** ❌

Your previous setup had the Codespaces config in `.devcontainer/codespaces/devcontainer.json`, which is not a standard location that Codespaces auto-detects.

The fix moves (creates) a proper config at the standard location while keeping the local config separate for VS Code local development.
