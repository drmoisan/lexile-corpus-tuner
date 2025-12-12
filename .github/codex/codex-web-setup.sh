#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== lexile-corpus-tuner setup: start ==="
echo "Working directory: $(pwd)"

REPO_ROOT="/workspace/lexile-corpus-tuner"

#
# 1. Python dependencies via Poetry (auto-detect groups)
#
if [ -f "$REPO_ROOT/pyproject.toml" ]; then
  echo "pyproject.toml found; installing Python dependencies with Poetry..."

  groups="$(poetry group list --format=plain 2>/dev/null || echo '')"
  echo "Poetry groups detected:"
  echo "${groups:-'(none)'}"

  if echo "$groups" | grep -qx 'dev' && echo "$groups" | grep -qx 'test'; then
    echo "Using groups: dev,test"
    poetry install --with dev,test
  elif echo "$groups" | grep -qx 'dev'; then
    echo "Using group: dev"
    poetry install --with dev
  else
    echo "No dev/test groups; installing default dependencies"
    poetry install
  fi
else
  echo "No pyproject.toml found; skipping Python dependency installation."
fi

#
# 2. Install PowerShell (pwsh) if missing
#
if ! command -v pwsh >/dev/null 2>&1; then
  echo "pwsh not found; installing PowerShell..."

  apt-get update -qq
  apt-get install -y --no-install-recommends \
    ca-certificates curl wget apt-transport-https gnupg

  POWERSHELL_DEB_URL="https://github.com/PowerShell/PowerShell/releases/download/v7.4.13/powershell_7.4.13-1.deb_amd64.deb"
  echo "Downloading PowerShell from: $POWERSHELL_DEB_URL"
  wget -qO /tmp/powershell.deb "$POWERSHELL_DEB_URL"

  echo "Installing PowerShell..."
  apt-get install -y /tmp/powershell.deb
  rm -f /tmp/powershell.deb
else
  echo "pwsh already installed; skipping PowerShell installation."
fi

#
# 3. Install PSScriptAnalyzer & Pester once, for ALL users (via PSGallery)
#    (this runs during setup, when internet is allowed)
#
if command -v pwsh >/dev/null 2>&1; then
  echo "PowerShell installed. Checking modules from PSGallery..."

  pwsh -NoLogo -NoProfile -Command '
    $ErrorActionPreference = "Stop"

    Write-Host "=== [ps] Checking PSScriptAnalyzer / Pester availability ==="

    # Make sure PSGallery is registered and trusted
    try {
      if (-not (Get-PSRepository -Name "PSGallery" -ErrorAction SilentlyContinue)) {
        Register-PSRepository -Default -ErrorAction SilentlyContinue
      }
    } catch {
      Write-Warning "Register-PSRepository -Default failed: $($_.Exception.Message)"
    }

    try {
      Set-PSRepository -Name "PSGallery" -InstallationPolicy Trusted -ErrorAction SilentlyContinue
    } catch {
      Write-Warning "Set-PSRepository PSGallery failed: $($_.Exception.Message)"
    }

    $modules = @("PSScriptAnalyzer", "Pester")

    foreach ($name in $modules) {
      $existing = Get-Module -ListAvailable -Name $name
      if ($existing) {
        Write-Host "$name already available: $($existing[0].Version) at $($existing[0].ModuleBase)"
        continue
      }

      Write-Host "$name not found; installing from PSGallery for AllUsers..."
      Install-Module -Name $name -Scope AllUsers -Force -SkipPublisherCheck -ErrorAction Stop
    }

    Write-Host "=== [ps] Final module list ==="
    Get-Module -ListAvailable PSScriptAnalyzer, Pester |
      Format-Table Name, Version, ModuleBase
  '

  #
  # 4. Optional: verify PoshQC is importable (non-fatal if missing)
  #
  POSHQC_PATH="$REPO_ROOT/scripts/powershell/PoshQC/PoshQC.psm1"
  if [ -f "$POSHQC_PATH" ]; then
    echo "Importing PoshQC module once in setup for sanity..."
    pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass \
      -Command '& { Import-Module "./scripts/powershell/PoshQC"; Get-Command -Module PoshQC | Out-Host }'
  else
    echo "PoshQC module not found at $POSHQC_PATH; skipping import."
  fi
else
  echo "pwsh is not available; skipping PowerShell tooling setup."
fi

echo "=== lexile-corpus-tuner setup: done ==="