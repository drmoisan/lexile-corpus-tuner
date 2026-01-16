# Use Python 3.13 base image with Debian Bookworm
FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm

# Install PowerShell
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
    # Dependencies for PowerShell and general tooling
    git \
    wget \
    apt-transport-https \
    software-properties-common \
    # Shell QC tooling
    shellcheck \
    shfmt \
    # Bash debugger build dependencies
    autoconf \
    automake \
    build-essential \
    texinfo \
    # Clipboard utilities for pyperclip fallback
    wl-clipboard \
    xclip \
    # Node tooling (required for Graphite CLI)
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Build and install bashdb (not packaged for all Debian/Ubuntu bases)
RUN git clone --depth 1 https://github.com/Trepan-Debuggers/bashdb /tmp/bashdb \
    && cd /tmp/bashdb \
    && bash ./autogen.sh \
    && bash ./configure \
    && make \
    && make install \
    && rm -rf /tmp/bashdb

# Install PowerShell 7.5+ from Microsoft repository
RUN wget -q https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y powershell \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==2.2.1

# Install Graphite CLI for the Graphite VS Code extension.
# Pinned for repeatable devcontainer builds.
ARG GRAPHITE_CLI_VERSION=1.7.14
RUN npm install -g "@withgraphite/graphite-cli@${GRAPHITE_CLI_VERSION}"

# Configure Poetry to create virtual environments in project directory
RUN poetry config virtualenvs.in-project true

# Set working directory
WORKDIR /workspaces/lexile-corpus-tuner

# Copy pyproject.toml and poetry.lock for dependency installation
# COPY pyproject.toml poetry.lock* ./

# Install Python dependencies (will be cached unless these files change)
# Note: This creates a basic venv; the post-create script will finalize it
# RUN poetry install --no-interaction --no-ansi --with dev || true

# Install PowerShell modules as root for all users
RUN pwsh -NoLogo -NoProfile -Command " \
    Set-PSRepository -Name PSGallery -InstallationPolicy Trusted; \
    Install-Module -Name PSScriptAnalyzer -Scope AllUsers -Force; \
    Install-Module -Name Pester -MinimumVersion 5.0.0 -Scope AllUsers -Force; \
    "

# Create non-root user (vscode is created by base image)
# Switch to vscode user
USER vscode

# Set up PowerShell profile location for vscode user
RUN mkdir -p ~/.config/powershell

# Install actionlint for GitHub Actions workflow linting (pin to latest)
USER root
RUN wget -q -O - https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash \
    | bash -s -- latest /usr/local/bin

# Install GitHub Copilot CLI (real agentic CLI, not the VS Code extension shim).
# Pinned to a known-good release for repeatable devcontainer builds.
ARG COPILOT_CLI_VERSION=0.0.377
RUN wget -qO- https://gh.io/copilot-install \
    | VERSION="${COPILOT_CLI_VERSION}" PREFIX="/usr/local" bash
USER vscode

# Set default shell to PowerShell for convenience
ENV SHELL=/usr/bin/pwsh

# The actual project files will be mounted at runtime
# Dependencies will be installed via postCreateCommand
