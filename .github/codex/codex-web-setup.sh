#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== lexile-corpus-tuner setup: start ==="
echo "Working directory: $(pwd)"

#
# 1. Python dependencies via Poetry (auto-detect groups)
#
if [ -f pyproject.toml ]; then
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
# 2. Install PowerShell (pwsh) on Ubuntu 24.04 base image, if missing
#
if ! command -v pwsh >/dev/null 2>&1; then
  echo "pwsh not found; installing PowerShell via .deb package..."

  # Install prerequisites
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    ca-certificates curl wget apt-transport-https gnupg

  # Use a specific LTS PowerShell build for Ubuntu (amd64)
  POWERSHELL_DEB_URL="https://github.com/PowerShell/PowerShell/releases/download/v7.4.13/powershell_7.4.13-1.deb_amd64.deb"

  echo "Downloading PowerShell from: $POWERSHELL_DEB_URL"
  wget -qO /tmp/powershell.deb "$POWERSHELL_DEB_URL"

  echo "Installing PowerShell..."
  apt-get install -y /tmp/powershell.deb

  rm -f /tmp/powershell.deb
else
  echo "pwsh already installed; skipping PowerShell installation."
fi

if command -v pwsh >/dev/null 2>&1; then
  echo "PowerShell installed. Version:"
  pwsh -NoLogo -NoProfile -Command '$PSVersionTable.PSVersion | Out-String'

  #
  # 3. Install PoshQC tools (your VS Code task, translated)
  #
  echo "Installing PoshQC tools via Install-PoshQCTools..."
  pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass \
    -Command '& { Import-Module "./scripts/powershell/PoshQC"; Install-PoshQCTools }'
  echo "PoshQC tools installation completed."
else
  echo "pwsh is still not available after attempted install; skipping PoshQC tooling."
fi

echo "=== lexile-corpus-tuner setup: done ==="