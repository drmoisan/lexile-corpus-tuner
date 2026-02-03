#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# codex-web-setup.sh
#
# Purpose:
#   Bootstrap the Codex web environment to match the local devcontainer tooling.
#   This ensures CI/Codex runs have parity with local development.
#
# Parity target: .devcontainer/local/Dockerfile + devcontainer.json
#
# Source Safety:
#   This script is safe to `source` for unit testing. All imperative execution
#   is wrapped in main() and guarded by BASH_SOURCE check at end of file.
# ------------------------------------------------------------------------------
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------------------
# Version pins (keep in sync with Dockerfile)
# ------------------------------------------------------------------------------
POETRY_VERSION="2.2.1"
PWSH_MIN_VERSION="7.4.0"
PWSH_FALLBACK_VERSION="7.4.13"
GRAPHITE_CLI_VERSION="1.7.14"
COPILOT_CLI_VERSION="0.0.377"
PSSA_VERSION="1.22.0"
PESTER_VERSION="5.6.1"

# ------------------------------------------------------------------------------
# Apt resilience configuration (defaults; override via environment)
# ------------------------------------------------------------------------------
APT_RETRY_ATTEMPTS_DEFAULT=5
APT_RETRY_DELAY_SECONDS_DEFAULT=5
APT_HTTP_TIMEOUT_SECONDS_DEFAULT=30
APT_DISABLE_PIPELINING_DEFAULT=1

APT_RETRY_ATTEMPTS="${APT_RETRY_ATTEMPTS:-$APT_RETRY_ATTEMPTS_DEFAULT}"
APT_RETRY_DELAY_SECONDS="${APT_RETRY_DELAY_SECONDS:-$APT_RETRY_DELAY_SECONDS_DEFAULT}"
APT_HTTP_TIMEOUT_SECONDS="${APT_HTTP_TIMEOUT_SECONDS:-$APT_HTTP_TIMEOUT_SECONDS_DEFAULT}"
APT_DISABLE_PIPELINING="${APT_DISABLE_PIPELINING:-$APT_DISABLE_PIPELINING_DEFAULT}"

# ------------------------------------------------------------------------------
# Function definitions (safe to source without side effects)
# ------------------------------------------------------------------------------

# Retry wrapper for apt operations to cope with transient 502/timeout issues.
# Logs: operation name (first argv token), attempt/max, proxy-env presence.
apt_with_retries() {
  local attempts="${APT_RETRY_ATTEMPTS}"
  local delay="${APT_RETRY_DELAY_SECONDS}"
  local operation="${1:-unknown}"
  local i=1

  # Log proxy environment presence (names only, not values)
  local proxy_info=""
  [ -n "${http_proxy:-}" ] && proxy_info="${proxy_info}http_proxy "
  [ -n "${https_proxy:-}" ] && proxy_info="${proxy_info}https_proxy "
  [ -n "${no_proxy:-}" ] && proxy_info="${proxy_info}no_proxy "
  if [ -n "$proxy_info" ]; then
    echo "[apt] Proxy env set: ${proxy_info}"
  fi

  while [ "$i" -le "$attempts" ]; do
    echo "[apt] ${operation}: attempt $i/$attempts"
    if "$@"; then
      return 0
    fi
    echo "[apt] Attempt $i/$attempts failed; retrying in ${delay}s..."
    sleep "$delay"
    i=$((i + 1))
  done
  echo "ERROR: apt command failed after $attempts attempts: $*" >&2
  return 1
}

apt_update() {
  local -a apt_args=(
    -o "Acquire::Retries=${APT_RETRY_ATTEMPTS}"
    -o "Acquire::http::Timeout=${APT_HTTP_TIMEOUT_SECONDS}"
    -o "Acquire::https::Timeout=${APT_HTTP_TIMEOUT_SECONDS}"
  )
  if [ "${APT_DISABLE_PIPELINING}" = "1" ]; then
    apt_args+=(-o "Acquire::http::Pipeline-Depth=0")
    apt_args+=(-o "Acquire::https::Pipeline-Depth=0")
  fi
  apt_with_retries apt-get "${apt_args[@]}" update -qq
}

apt_install() {
  local -a apt_args=(
    -o "Acquire::Retries=${APT_RETRY_ATTEMPTS}"
    -o "Acquire::http::Timeout=${APT_HTTP_TIMEOUT_SECONDS}"
    -o "Acquire::https::Timeout=${APT_HTTP_TIMEOUT_SECONDS}"
  )
  if [ "${APT_DISABLE_PIPELINING}" = "1" ]; then
    apt_args+=(-o "Acquire::http::Pipeline-Depth=0")
    apt_args+=(-o "Acquire::https::Pipeline-Depth=0")
  fi
  apt_with_retries apt-get "${apt_args[@]}" install -y --no-install-recommends --fix-missing "$@"
}

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

# ------------------------------------------------------------------------------
# 0. System packages (match Dockerfile apt-get install)
# ------------------------------------------------------------------------------
install_system_packages() {
  echo "Installing system packages for devcontainer parity..."

  apt_update
  apt_install \
    git \
    wget \
    curl \
    apt-transport-https \
    software-properties-common \
    ca-certificates \
    gnupg \
    shellcheck \
    shfmt \
    autoconf \
    automake \
    build-essential \
    texinfo \
    wl-clipboard \
    xclip \
    nodejs \
    npm

  # Install bats-core for shell testing (developer-tooling.md mentions it)
  if ! command -v bats >/dev/null 2>&1; then
    echo "Installing bats-core for shell tests..."
    npm install -g bats || echo "WARN: bats-core install failed; shell tests may be skipped"
  fi
}

# Post-install validation: fail if required executables are missing.
validate_required_tools() {
  local required_tools=("shellcheck" "shfmt" "node" "npm")
  local missing=()

  for tool in "${required_tools[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      missing+=("$tool")
    fi
  done

  if [ ${#missing[@]} -gt 0 ]; then
    for tool in "${missing[@]}"; do
      echo "ERROR: missing required tool: $tool" >&2
    done
    return 1
  fi

  echo "All required tools validated: ${required_tools[*]}"
  return 0
}

# ------------------------------------------------------------------------------
# 0b. Build bashdb from source (not packaged for all Debian/Ubuntu bases)
# ------------------------------------------------------------------------------
install_bashdb() {
  if command -v bashdb >/dev/null 2>&1; then
    echo "bashdb already installed"
    return 0
  fi

  echo "Building bashdb from source..."
  local tmpdir
  tmpdir="$(mktemp -d)"
  git clone --depth 1 https://github.com/Trepan-Debuggers/bashdb "$tmpdir/bashdb"
  cd "$tmpdir/bashdb"
  bash ./autogen.sh
  bash ./configure
  make
  make install
  cd -
  rm -rf "$tmpdir"
  echo "bashdb installed successfully"
}

# Retry wrapper to cope with transient network/DNS hiccups when hitting PyPI
install_with_retries() {
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

# ------------------------------------------------------------------------------
# 2. Install/upgrade PowerShell with distro-aware selection (fall back to GitHub .deb)
# ------------------------------------------------------------------------------
ensure_pwsh_system() {
  # Support env overrides for OS detection (used in tests)
  local os_id="" os_version="" repo_url="" fallback_version="${PWSH_FALLBACK_VERSION}"
  if [ -n "${CODEX_OS_ID:-}" ]; then
    os_id="${CODEX_OS_ID}"
  elif [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    os_id="${ID:-}"
  fi

  if [ -n "${CODEX_OS_VERSION:-}" ]; then
    os_version="${CODEX_OS_VERSION}"
  elif [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    os_version="${VERSION_ID:-}"
  fi

  # Choose the correct Microsoft feed for the host; Ubuntu images were failing with the Debian feed.
  if [[ "$os_id" == "ubuntu" || "$os_id" == "debian" ]]; then
    repo_url="https://packages.microsoft.com/config/${os_id}/${os_version:-12}/packages-microsoft-prod.deb"
  else
    repo_url="https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb"
  fi

  apt_update
  apt_install \
    ca-certificates curl wget apt-transport-https gnupg software-properties-common

  local repo_ok=0
  if wget -q "$repo_url" -O /tmp/packages-microsoft-prod.deb; then
    if dpkg -i /tmp/packages-microsoft-prod.deb; then
      repo_ok=1
    fi
  fi
  rm -f /tmp/packages-microsoft-prod.deb

  if [ "$repo_ok" -eq 1 ]; then
    apt_update
    if apt_install powershell; then
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
  apt_install /tmp/powershell.deb
  rm -f /tmp/powershell.deb
}

install_pwsh_user() {
  local version="${PWSH_FALLBACK_VERSION}"
  local arch
  arch="$(uname -m)"
  local pwsh_arch=""
  case "$arch" in
    x86_64|amd64)
      pwsh_arch="linux-x64"
      ;;
    aarch64|arm64)
      pwsh_arch="linux-arm64"
      ;;
    *)
      echo "ERROR: unsupported architecture for PowerShell user install: ${arch}" >&2
      return 1
      ;;
  esac

  local archive="powershell-${version}-${pwsh_arch}.tar.gz"
  local url="https://github.com/PowerShell/PowerShell/releases/download/v${version}/${archive}"
  local target_dir="${HOME}/.local/pwsh"
  local bin_dir="${HOME}/.local/bin"
  local tmpdir

  echo "Installing pwsh ${version} to ${target_dir} (user-local, no sudo)..."
  mkdir -p "$target_dir" "$bin_dir"
  tmpdir="$(mktemp -d)"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$tmpdir/$archive"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$tmpdir/$archive" "$url"
  else
    echo "ERROR: neither curl nor wget is available to download PowerShell." >&2
    rm -rf "$tmpdir"
    return 1
  fi

  tar -xzf "$tmpdir/$archive" -C "$target_dir"
  rm -rf "$tmpdir"

  if [ ! -x "$target_dir/pwsh" ]; then
    echo "ERROR: pwsh binary not found after extraction to ${target_dir}." >&2
    return 1
  fi

  ln -sf "$target_dir/pwsh" "$bin_dir/pwsh"
  export PATH="$bin_dir:$PATH"
  echo "pwsh installed to ${target_dir} and linked in ${bin_dir}."
  echo "Tip: add \"export PATH=\"${bin_dir}:\$PATH\"\" to your shell profile for persistence."
  return 0
}

ensure_pwsh() {
  # Keep requirement aligned with devcontainer but allow the known-good 7.4.x fallback
  local required="${PWSH_MIN_VERSION}"

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

  if [ "$(id -u)" -eq 0 ]; then
    ensure_pwsh_system
  elif command -v sudo >/dev/null 2>&1; then
    sudo bash -c "
      PWSH_MIN_VERSION='${PWSH_MIN_VERSION}'
      PWSH_FALLBACK_VERSION='${PWSH_FALLBACK_VERSION}'
      APT_RETRY_ATTEMPTS='${APT_RETRY_ATTEMPTS}'
      APT_RETRY_DELAY_SECONDS='${APT_RETRY_DELAY_SECONDS}'
      APT_HTTP_TIMEOUT_SECONDS='${APT_HTTP_TIMEOUT_SECONDS}'
      APT_DISABLE_PIPELINING='${APT_DISABLE_PIPELINING}'
      $(declare -f apt_with_retries apt_update apt_install ensure_pwsh_system)
      ensure_pwsh_system
    "
  else
    install_pwsh_user
  fi
}

# ------------------------------------------------------------------------------
# 5. Install Graphite CLI (required for Graphite VS Code extension parity)
# ------------------------------------------------------------------------------
install_graphite_cli() {
  if command -v gt >/dev/null 2>&1; then
    echo "Graphite CLI already installed"
    return 0
  fi

  if ! command -v npm >/dev/null 2>&1; then
    echo "WARN: npm not available; skipping Graphite CLI installation"
    return 0
  fi

  echo "Installing Graphite CLI v${GRAPHITE_CLI_VERSION}..."
  npm install -g "@withgraphite/graphite-cli@${GRAPHITE_CLI_VERSION}" || {
    echo "WARN: Graphite CLI install failed; non-blocking"
  }
}

# ------------------------------------------------------------------------------
# 6. Install GitHub Copilot CLI (real agentic CLI, not VS Code extension shim)
# ------------------------------------------------------------------------------
install_copilot_cli() {
  if command -v github-copilot-cli >/dev/null 2>&1; then
    echo "GitHub Copilot CLI already installed"
    return 0
  fi

  echo "Installing GitHub Copilot CLI v${COPILOT_CLI_VERSION}..."
  local install_prefix="/usr/local"
  local user_prefix="${HOME}/.local"

  # Use sudo if available and not root
  if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    wget -qO- https://gh.io/copilot-install | sudo VERSION="${COPILOT_CLI_VERSION}" PREFIX="${install_prefix}" bash || {
      echo "WARN: GitHub Copilot CLI install failed; non-blocking"
    }
  elif [ "$(id -u)" -eq 0 ]; then
    wget -qO- https://gh.io/copilot-install | VERSION="${COPILOT_CLI_VERSION}" PREFIX="${install_prefix}" bash || {
      echo "WARN: GitHub Copilot CLI install failed; non-blocking"
    }
  else
    echo "No root/sudo access; installing Copilot CLI to ${user_prefix}"
    mkdir -p "${user_prefix}/bin"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL https://gh.io/copilot-install | VERSION="${COPILOT_CLI_VERSION}" PREFIX="${user_prefix}" bash || {
        echo "WARN: GitHub Copilot CLI install failed; non-blocking"
      }
    else
      wget -qO- https://gh.io/copilot-install | VERSION="${COPILOT_CLI_VERSION}" PREFIX="${user_prefix}" bash || {
        echo "WARN: GitHub Copilot CLI install failed; non-blocking"
      }
    fi
    export PATH="${user_prefix}/bin:$PATH"
  fi
}

# ------------------------------------------------------------------------------
# 4. Install actionlint (user-local fallback when sudo is unavailable)
# ------------------------------------------------------------------------------
install_actionlint() {
  if command -v actionlint >/dev/null 2>&1; then
    echo "actionlint already installed"
    return 0
  fi

  echo "Installing actionlint..."
  if [ "$(id -u)" -eq 0 ] || command -v sudo >/dev/null 2>&1; then
    if [ "$(id -u)" -ne 0 ]; then
      sudo bash -c 'wget -q -O - https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash | bash -s -- latest /usr/local/bin'
    else
      wget -q -O - https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash | bash -s -- latest /usr/local/bin
    fi
    return 0
  fi

  local user_bin="${HOME}/.local/bin"
  mkdir -p "${user_bin}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash | bash -s -- latest "${user_bin}"
  else
    wget -q -O - https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash | bash -s -- latest "${user_bin}"
  fi
  export PATH="${user_bin}:$PATH"
}

# ------------------------------------------------------------------------------
# Main entry point (guarded for source safety)
# ------------------------------------------------------------------------------
main() {
  echo "=== lexile-corpus-tuner setup: start ==="
  echo "Working directory: $(pwd)"

  REPO_ROOT="${WORKSPACE_FOLDER:-}"
  if [ -z "$REPO_ROOT" ] || [ ! -d "$REPO_ROOT" ]; then
    REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  fi

  # Normalize to an absolute path
  REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
  export REPO_ROOT

  # --------------------------------------------------------------------------
  # 0. System packages
  # --------------------------------------------------------------------------
  if [ "$(id -u)" -eq 0 ] || command -v sudo >/dev/null 2>&1; then
    if [ "$(id -u)" -ne 0 ]; then
      # Re-exec with sudo for apt operations; pass helper functions and APT_* values
      sudo bash -c "
        APT_RETRY_ATTEMPTS='${APT_RETRY_ATTEMPTS}'
        APT_RETRY_DELAY_SECONDS='${APT_RETRY_DELAY_SECONDS}'
        APT_HTTP_TIMEOUT_SECONDS='${APT_HTTP_TIMEOUT_SECONDS}'
        APT_DISABLE_PIPELINING='${APT_DISABLE_PIPELINING}'
        $(declare -f apt_with_retries apt_update apt_install install_system_packages)
        install_system_packages
      "
    else
      install_system_packages
    fi
  else
    echo "WARN: No root/sudo access; skipping system package installation"
  fi

  # Validate required tools are present after system package install
  validate_required_tools

  # --------------------------------------------------------------------------
  # 0b. bashdb from source
  # --------------------------------------------------------------------------
  if [ "$(id -u)" -eq 0 ]; then
    install_bashdb
  elif command -v sudo >/dev/null 2>&1; then
    sudo bash -c "$(declare -f install_bashdb); install_bashdb"
  else
    echo "WARN: No root/sudo access; skipping bashdb installation"
  fi

  # --------------------------------------------------------------------------
  # 1. Ensure Poetry is available
  # --------------------------------------------------------------------------
  if command -v poetry >/dev/null 2>&1; then
    echo "Poetry present ($(poetry --version)); reusing existing installation."
  else
    echo "Poetry not found; installing Poetry ${POETRY_VERSION} (devcontainer baseline)..."
    pip install --no-cache-dir "poetry==${POETRY_VERSION}"
  fi

  # --------------------------------------------------------------------------
  # 1b. Python dependencies via Poetry
  # --------------------------------------------------------------------------
  cd "$REPO_ROOT"
  poetry config virtualenvs.in-project true --local

  # If a custom index is provided, surface it before installs
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

  # --------------------------------------------------------------------------
  # 2. PowerShell
  # --------------------------------------------------------------------------
  ensure_pwsh

  # --------------------------------------------------------------------------
  # 3. Install PSScriptAnalyzer & Pester
  # --------------------------------------------------------------------------
  if command -v pwsh >/dev/null 2>&1; then
    echo "PowerShell installed. Checking modules from PSGallery..."

    pwsh -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command '
      $ErrorActionPreference = "Stop"
      $ProgressPreference = "SilentlyContinue"
      [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

      Write-Host "=== [ps] Checking PSScriptAnalyzer / Pester availability ==="

      try {
        if (-not (Get-PackageProvider -Name NuGet -ErrorAction SilentlyContinue)) {
          Install-PackageProvider -Name NuGet -Scope CurrentUser -Force -ErrorAction Stop
        }
      } catch {
        Write-Warning "Install-PackageProvider NuGet failed: $($_.Exception.Message)"
      }

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
        @{ Name = "PSScriptAnalyzer"; Version = "'"${PSSA_VERSION}"'" },
        @{ Name = "Pester"; Version = "'"${PESTER_VERSION}"'" }
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
        Install-Module -Name $module.Name -RequiredVersion $module.Version -Repository PSGallery -Scope CurrentUser -AllowClobber -Force -Confirm:$false -ErrorAction Stop
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

  # --------------------------------------------------------------------------
  # 4. Install actionlint
  # --------------------------------------------------------------------------
  install_actionlint

  # --------------------------------------------------------------------------
  # 5. Graphite CLI
  # --------------------------------------------------------------------------
  install_graphite_cli

  # --------------------------------------------------------------------------
  # 6. GitHub Copilot CLI
  # --------------------------------------------------------------------------
  install_copilot_cli

  echo "=== lexile-corpus-tuner setup: done ==="
}

# Source guard: only run main() when executed directly, not when sourced
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
