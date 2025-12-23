#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== lexile-corpus-tuner setup: start ==="
echo "Working directory: $(pwd)"

REPO_ROOT="${WORKSPACE_FOLDER:-}"
if [ -z "$REPO_ROOT" ] || [ ! -d "$REPO_ROOT" ]; then
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

# Normalize to an absolute path
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
export REPO_ROOT

# Quick connectivity preflight to avoid long retries when PyPI is unreachable.
check_pypi_connectivity() {
  if [ "${ALLOW_OFFLINE_INSTALL:-0}" = "1" ]; then
    echo "ALLOW_OFFLINE_INSTALL=1 set; skipping PyPI connectivity check."
    return 0
  fi

  if curl -I -s --max-time 5 https://pypi.org/simple >/dev/null; then
    return 0
  fi

  echo "ERROR: Unable to reach pypi.org (check network/DNS or set ALLOW_OFFLINE_INSTALL=1 to skip)." >&2
  echo "Tip: set POETRY_PYPI_URL to a reachable mirror if direct PyPI access is blocked." >&2
  return 1
}

#
# 0. Ensure Poetry is available (reuse existing install to avoid extra PyPI traffic); prefer 2.2.1 if absent
#
if command -v poetry >/dev/null 2>&1; then
  echo "Poetry present ($(poetry --version)); reusing existing installation."
else
  echo "Poetry not found; installing Poetry 2.2.1 (devcontainer baseline)..."
  pip install --no-cache-dir "poetry==2.2.1"
fi

#
# 1. Python dependencies via Poetry (devcontainer parity: in-project venv, --with dev)
#
cd "$REPO_ROOT"
poetry config virtualenvs.in-project true --local

# If a custom index is provided, surface it before installs so we know which endpoint Poetry will hit.
if [ -n "${POETRY_PYPI_URL:-}" ]; then
  echo "Using custom POETRY_PYPI_URL=${POETRY_PYPI_URL}"
  poetry config repositories.main "$POETRY_PYPI_URL"
  poetry config pypi-token.main "" 2>/dev/null || true
fi

# Bail out early if PyPI is unreachable (unless ALLOW_OFFLINE_INSTALL=1)
check_pypi_connectivity

if [ -d ".venv" ] && [ ! -x ".venv/bin/python" ]; then
  echo "Detected broken .venv; removing and recreating..."
  rm -rf .venv
fi

install_with_retries() {
  # Retry wrapper to cope with transient network/DNS hiccups when hitting PyPI
  local attempts=5
  local delay=5
  local i=1
  while [ "$i" -le "$attempts" ]; do
    if "$@"; then
      return 0
    fi
    echo "Attempt $i/$attempts failed; retrying in ${delay}s..."
    sleep "$delay"
    i=$((i + 1))
  done
  echo "ERROR: command failed after $attempts attempts: $*" >&2
  return 1
}

if [ -f "poetry.lock" ]; then
  echo "poetry.lock found; installing locked dependencies with --with dev..."
  install_with_retries poetry install --no-interaction --no-ansi --with dev
elif [ -f "pyproject.toml" ]; then
  echo "poetry.lock missing; locking and installing with --with dev..."
  install_with_retries poetry lock --no-interaction --no-ansi
  install_with_retries poetry install --no-interaction --no-ansi --with dev
else
  echo "No pyproject.toml found; skipping Python dependency installation."
fi

#
# 2. Install/upgrade PowerShell with distro-aware selection (fall back to GitHub .deb)
ensure_pwsh() {
  # Keep requirement aligned with devcontainer but allow the known-good 7.4.x fallback
  local required="7.4.0"

  if command -v pwsh >/dev/null 2>&1; then
    local current
    current="$(pwsh --version | awk '{print $2}')"
    if dpkg --compare-versions "$current" ge "$required"; then
      echo "pwsh $current present; meets requirement ($required)."
      return 0
    fi
    echo "pwsh $current present; upgrading to >= $required..."
  else
    echo "pwsh not found; installing PowerShell..."
  fi

  local os_id="" os_version="" repo_url="" fallback_version="7.4.13"
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    os_id="${ID:-}"
    os_version="${VERSION_ID:-}"
  fi

  # Choose the correct Microsoft feed for the host; Ubuntu images were failing with the Debian feed.
  if [[ "$os_id" == "ubuntu" || "$os_id" == "debian" ]]; then
    repo_url="https://packages.microsoft.com/config/${os_id}/${os_version:-12}/packages-microsoft-prod.deb"
  else
    repo_url="https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb"
  fi

  apt-get update -qq
  apt-get install -y --no-install-recommends \
    ca-certificates curl wget apt-transport-https gnupg software-properties-common

  local repo_ok=0
  if wget -q "$repo_url" -O /tmp/packages-microsoft-prod.deb; then
    if dpkg -i /tmp/packages-microsoft-prod.deb; then
      repo_ok=1
    fi
  fi
  rm -f /tmp/packages-microsoft-prod.deb

  if [ "$repo_ok" -eq 1 ]; then
    apt-get update -qq
    if apt-get install -y --no-install-recommends powershell; then
      return 0
    fi
    echo "PowerShell install from distro feed failed; falling back to GitHub package." >&2
  else
    echo "PowerShell feed bootstrap failed for ${os_id:-unknown}; falling back to GitHub package." >&2
  fi

  local pwsh_deb="powershell_${fallback_version}-1.deb_amd64.deb"
  local pwsh_url="https://github.com/PowerShell/PowerShell/releases/download/v${fallback_version}/${pwsh_deb}"
  echo "Downloading PowerShell fallback from: ${pwsh_url}"
  wget -qO /tmp/powershell.deb "$pwsh_url"
  apt-get install -y /tmp/powershell.deb
  rm -f /tmp/powershell.deb
}

ensure_pwsh

#
# 3. Install PSScriptAnalyzer & Pester (pinned to devcontainer minimums)
if command -v pwsh >/dev/null 2>&1; then
  echo "PowerShell installed. Checking modules from PSGallery..."

  pwsh -NoLogo -NoProfile -Command '
    $ErrorActionPreference = "Stop"

    Write-Host "=== [ps] Checking PSScriptAnalyzer / Pester availability ==="

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

    $required = @(
      @{ Name = "PSScriptAnalyzer"; Version = "1.22.0" },
      @{ Name = "Pester"; Version = "5.6.1" }
    )

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

    Write-Host "=== [ps] Final module list ==="
    Get-Module -ListAvailable PSScriptAnalyzer, Pester |
      Sort-Object Name, Version -Descending |
      Format-Table Name, Version, ModuleBase
  '

  POSHQC_PATH="$REPO_ROOT/scripts/powershell/PoshQC/PoshQC.psd1"
  if [ -f "$POSHQC_PATH" ]; then
    echo "Importing PoshQC module (required for parity)..."
    pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass \
      -Command '& { Import-Module "$env:REPO_ROOT/scripts/powershell/PoshQC/PoshQC.psd1" -Force; Get-Command -Module PoshQC | Out-Host }'
  else
    echo "ERROR: PoshQC module not found at $POSHQC_PATH" >&2
    exit 1
  fi
else
  echo "pwsh is not available; skipping PowerShell tooling setup."
fi

# 4. Install actionlint to match devcontainer tooling
if ! command -v actionlint >/dev/null 2>&1; then
  echo "Installing actionlint..."
  wget -q -O - https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash | bash -s -- latest /usr/local/bin
else
  echo "actionlint already installed"
fi

echo "=== lexile-corpus-tuner setup: done ==="