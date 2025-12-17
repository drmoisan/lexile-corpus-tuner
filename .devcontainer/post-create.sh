#!/bin/bash
set -e

echo "==================================="
echo "Post-Create Container Setup"
echo "==================================="

# -----------------------------------------------------------------------------
# Resolve workspace/repo root deterministically
# -----------------------------------------------------------------------------
# Prefer devcontainer-provided workspace folder when available; fall back to git;
# finally fall back to current directory.
WORKSPACE_DIR="${WORKSPACE_FOLDER:-}"
if [ -n "$WORKSPACE_DIR" ] && [ -d "$WORKSPACE_DIR" ]; then
  # If WORKSPACE_DIR is not a git repo root, try to resolve to the git top-level within it
  if git -C "$WORKSPACE_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    WORKSPACE_DIR="$(git -C "$WORKSPACE_DIR" rev-parse --show-toplevel)"
  fi
else
  WORKSPACE_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

cd "$WORKSPACE_DIR"
export WORKSPACE_DIR
echo "Workspace directory: $WORKSPACE_DIR"

# Create PowerShell profile with custom prompt
echo ""
echo "Creating PowerShell profile..."
mkdir -p ~/.config/powershell
cat > ~/.config/powershell/Microsoft.PowerShell_profile.ps1 << 'PROFILE_END'
$env:WORKSPACE_DIR = '/workspaces/lexile-corpus-tuner'

# Activate venv silently (suppress all output)
$activateScript = "$env:WORKSPACE_DIR/.venv/bin/Activate.ps1"
if (Test-Path $activateScript) {
    $null = & $activateScript *>&1
}

# Custom prompt with relative paths
function prompt {
    $venvName = if ($env:VIRTUAL_ENV) {
        $name = Split-Path -Leaf $env:VIRTUAL_ENV
        if ($name -eq '.venv' -and (Test-Path "$env:VIRTUAL_ENV/pyvenv.cfg")) {
            $cfg = Get-Content "$env:VIRTUAL_ENV/pyvenv.cfg" -Raw
            if ($cfg -match 'prompt\s*=\s*(.+)') {
                $matches[1].Trim()
            } else {
                '.venv'
            }
        } else {
            $name
        }
    } else {
        ''
    }
    
    $venv = if ($venvName) { "($venvName)" } else { "" }
    $currentPath = $PWD.Path
    $workspaceDir = $env:WORKSPACE_DIR
    
    if ($currentPath -eq $workspaceDir) {
        $relativePath = "/"
    } elseif ($currentPath.StartsWith($workspaceDir)) {
        $relativePath = $currentPath.Substring($workspaceDir.Length)
    } else {
        $relativePath = $currentPath
    }
    
    "$venv$relativePath> "
}
PROFILE_END


# Persist WORKSPACE_DIR for future bash sessions (optional)
BASHRC="$HOME/.bashrc"
LINE="export WORKSPACE_DIR=\"$WORKSPACE_DIR\""
grep -qxF "$LINE" "$BASHRC" 2>/dev/null || echo "$LINE" >> "$BASHRC"

# -----------------------------------------------------------------------------
# Python / Poetry setup
# -----------------------------------------------------------------------------
echo ""
echo "Installing Python dependencies with Poetry..."

# Ensure Poetry uses an in-project venv for THIS repo (local config, not global)
poetry config virtualenvs.in-project true --local

# If the venv exists but is clearly broken, recreate it
if [ -d ".venv" ] && [ ! -x ".venv/bin/python" ]; then
  echo "Detected broken .venv; removing and recreating..."
  rm -rf .venv
fi

if [ -f "poetry.lock" ]; then
  echo "poetry.lock found; installing locked dependencies..."
  poetry install --no-interaction --no-ansi --with dev
else
  echo "Warning: poetry.lock not found. Resolving dependencies and creating lock file..."
  poetry lock --no-interaction --no-ansi
  poetry install --no-interaction --no-ansi --with dev
fi

# Verify Poetry is using the in-project environment (warn if not)
echo ""
echo "Verifying Poetry environment..."
ENV_PATH="$(poetry env info --path 2>/dev/null || true)"
if [ -z "$ENV_PATH" ]; then
  echo "Warning: could not determine Poetry env path (poetry env info --path returned nothing)."
elif [ "$ENV_PATH" != "$WORKSPACE_DIR/.venv" ]; then
  echo "Warning: Poetry env is '$ENV_PATH' (expected '$WORKSPACE_DIR/.venv')."
else
  echo "Poetry env path: $ENV_PATH"
fi

# Verify Python tooling
echo ""
echo "Verifying Python tooling..."
poetry run black --version
poetry run ruff --version
poetry run pyright --version
poetry run pytest --version

# -----------------------------------------------------------------------------
# PowerShell tooling verification
# -----------------------------------------------------------------------------
echo ""
echo "Verifying PowerShell tooling..."
pwsh -NoLogo -NoProfile -Command "
    Write-Host 'PowerShell version:'
    \$PSVersionTable.PSVersion
    Write-Host ''
    Write-Host 'Installed modules:'
    Get-Module -ListAvailable PSScriptAnalyzer, Pester | Sort-Object Name, Version -Descending | Format-Table Name, Version
"

# -----------------------------------------------------------------------------
# PowerShell QC dependencies
# -----------------------------------------------------------------------------
# Recommendation: Choose ONE strategy for module installation (AllUsers pinned in Dockerfile
# OR CurrentUser pinned here). This script keeps the original behavior (CurrentUser pinning)
# but will not reinstall if a suitable version is already present.
echo ""
echo "Ensuring PowerShell QC dependencies..."
pwsh -NoLogo -NoProfile -Command - <<'PWSH'
$ErrorActionPreference = 'Stop'
$required = @(
    @{ Name = 'PSScriptAnalyzer'; Version = '1.22.0' },
    @{ Name = 'Pester'; Version = '5.6.1' }
)

$gallery = Get-PSRepository -Name PSGallery -ErrorAction SilentlyContinue
if (-not $gallery) {
    Register-PSRepository -Default -InstallationPolicy Trusted
} elseif ($gallery.InstallationPolicy -ne 'Trusted') {
    try {
        Set-PSRepository -Name PSGallery -InstallationPolicy Trusted -ErrorAction Stop
    } catch {
        Write-Warning 'Could not set PSGallery to Trusted automatically; you may see prompts during install.'
    }
}

foreach ($module in $required) {
    $installed = Get-Module -ListAvailable -Name $module.Name |
        Where-Object { $_.Version -ge [version]$module.Version } |
        Sort-Object Version -Descending |
        Select-Object -First 1

    if ($installed) {
        Write-Host "Found $($module.Name) $($installed.Version)"
        continue
    }

    Write-Host "Installing $($module.Name) $($module.Version) (CurrentUser scope)..."
    Install-Module -Name $module.Name -RequiredVersion $module.Version -Scope CurrentUser -AllowClobber -Force -ErrorAction Stop
}
PWSH

# -----------------------------------------------------------------------------
# Import PoshQC module from the mounted workspace
# -----------------------------------------------------------------------------
echo ""
echo "Importing PoshQC module..."
pwsh -NoLogo -NoProfile -Command - <<'PWSH'
$ErrorActionPreference = 'Stop'

$candidate = "$env:WORKSPACE_DIR/scripts/powershell/PoshQC/PoshQC.psd1"
if (-not (Test-Path $candidate)) {
    Write-Error "PoshQC module not found. Checked: $candidate"
    exit 1
}

Import-Module $candidate -Force -ErrorAction Stop
Write-Host "Imported PoshQC from $candidate"
PWSH

# -----------------------------------------------------------------------------
# Git configuration reminder
# -----------------------------------------------------------------------------
if [ ! -f ~/.gitconfig ]; then
  echo ""
  echo "Git configuration not found. You may want to run:"
  echo "  git config --global user.name 'Your Name'"
  echo "  git config --global user.email 'your.email@example.com'"
fi

# -----------------------------------------------------------------------------
# Completion banner
# -----------------------------------------------------------------------------
echo ""
echo "==================================="
echo "Dev Container Setup Complete!"
echo "==================================="
echo ""
echo "Available commands:"
echo "  poetry run pytest                         # Run Python tests"
echo "  poetry run black .                        # Format Python code"
echo "  poetry run ruff check                     # Lint Python code"
echo "  poetry run pyright                        # Type check Python code"
echo "  pwsh -File scripts/dev-tools/fix-all.ps1  # Run all QC checks"
echo ""
echo "VS Code tasks are available via:"
echo "  Ctrl+Shift+P -> Tasks: Run Task"
echo ""
echo "To build the package:"
echo "  poetry build"
echo ""
echo "To run the CLI:"
echo "  poetry run lexile-tuner --help"
echo "  poetry run lexile-scoring-model-pipeline --help"
echo ""