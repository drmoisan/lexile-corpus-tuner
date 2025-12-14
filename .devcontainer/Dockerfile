# Use Python 3.13 base image with Debian Bookworm
FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm

# Install PowerShell
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
    # Dependencies for PowerShell
    wget \
    apt-transport-https \
    software-properties-common \
    # cloc for code metrics
    cloc \
    && rm -rf /var/lib/apt/lists/*

# Install PowerShell 7.5+ from Microsoft repository
RUN wget -q https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y powershell \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Configure Poetry to create virtual environments in project directory
RUN poetry config virtualenvs.in-project true

# Set working directory
WORKDIR /workspace

# Copy pyproject.toml and poetry.lock for dependency installation
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies (will be cached unless these files change)
# Note: This creates a basic venv; the post-create script will finalize it
RUN poetry install --no-interaction --no-ansi --with dev || true

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

# Install actionlint for GitHub Actions workflow linting
USER root
RUN wget -q -O - https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash | bash -s -- -d /usr/local/bin
USER vscode

# Set default shell to PowerShell for convenience
ENV SHELL=/usr/bin/pwsh

# The actual project files will be mounted at runtime
# Dependencies will be installed via postCreateCommand
