#!/bin/bash
set -e

echo "==================================="
echo "Post-Create Container Setup"
echo "==================================="

# Ensure we're in the workspace directory
cd /workspace

# Install/update Python dependencies with Poetry
echo "Installing Python dependencies with Poetry..."
if [ -f "poetry.lock" ]; then
    poetry install --no-interaction --no-ansi --with dev
else
    echo "Warning: poetry.lock not found. Creating from pyproject.toml..."
    poetry install --no-interaction --no-ansi --with dev
fi

# Verify Python tooling
echo ""
echo "Verifying Python tooling..."
poetry run black --version
poetry run ruff --version
poetry run pyright --version
poetry run pytest --version

# Verify PowerShell tooling
echo ""
echo "Verifying PowerShell tooling..."
pwsh -NoLogo -NoProfile -Command "
    Write-Host 'PowerShell version:'
    \$PSVersionTable.PSVersion
    Write-Host ''
    Write-Host 'Installed modules:'
    Get-Module -ListAvailable PSScriptAnalyzer, Pester | Format-Table Name, Version
"

# Ensure PowerShell QC dependencies are present before importing PoshQC
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
    $installed = Get-Module -ListAvailable -Name $module.Name | Where-Object { $_.Version -ge [version]$module.Version } | Select-Object -First 1
    if ($installed) {
        Write-Host "Found $($module.Name) $($installed.Version)"
        continue
    }

    Write-Host "Installing $($module.Name) $($module.Version) (CurrentUser scope)..."
    Install-Module -Name $module.Name -RequiredVersion $module.Version -Scope CurrentUser -AllowClobber -Force -ErrorAction Stop
}
PWSH

# Import PoshQC module (will be available from mounted workspace)
echo ""
echo "Importing PoshQC module..."
pwsh -NoLogo -NoProfile -Command - <<'PWSH'
$ErrorActionPreference = 'Stop'
$candidates = @(
    '/workspace/scripts/powershell/PoshQC/PoshQC.psd1',
    '/workspaces/lexile-corpus-tuner/scripts/powershell/PoshQC/PoshQC.psd1'
)

$modulePath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $modulePath) {
    Write-Error "PoshQC module not found. Checked: $($candidates -join ', ')"
    exit 1
}

Import-Module $modulePath -Force -ErrorAction Stop
Write-Host "Imported PoshQC from $modulePath"
PWSH

# Set up git configuration (if not already configured)
if [ ! -f ~/.gitconfig ]; then
    echo ""
    echo "Git configuration not found. You may want to run:"
    echo "  git config --global user.name 'Your Name'"
    echo "  git config --global user.email 'your.email@example.com'"
fi

# Display helpful information
echo ""
echo "==================================="
echo "Dev Container Setup Complete!"
echo "==================================="
echo ""
echo "Available commands:"
echo "  poetry run pytest                    # Run Python tests"
echo "  poetry run black .                   # Format Python code"
echo "  poetry run ruff check                # Lint Python code"
echo "  poetry run pyright                   # Type check Python code"
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
