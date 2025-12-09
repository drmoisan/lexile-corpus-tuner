$script:ModuleRoot = Split-Path -Parent $PSCommandPath
$script:PssaSettings = Join-Path $ModuleRoot 'settings/pssa.settings.psd1'
$script:PesterSettings = Join-Path $ModuleRoot 'settings/pester.runsettings.psd1'

$script:DefaultExcludedDirs = @(
    '.git', '.venv', 'venv', 'node_modules', 'dist', 'build', '.pytest_cache',
    '__pycache__', '.mypy_cache', '.ruff_cache', '.vscode', '.idea', 'artifacts'
)

function Get-PoshQCFileList {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Root,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs
    )

    $resolvedRoot = Resolve-Path -Path $Root -ErrorAction Stop
    $files = Get-ChildItem -Path $resolvedRoot -Recurse -Include *.ps1, *.psm1, *.psd1 -File
    if (-not $files) { return @() }

    $result = @()
    foreach ($file in $files) {
        $parts = $file.FullName.Split([IO.Path]::DirectorySeparatorChar, [System.StringSplitOptions]::RemoveEmptyEntries)
        $skip = $false
        foreach ($dir in $ExcludeDirs) {
            if ($parts -contains $dir) { $skip = $true; break }
        }
        if (-not $skip) { $result += $file }
    }
    return $result
}

function Install-PoshQCTools {
    [CmdletBinding()]
    param()

    $ErrorActionPreference = 'Stop'

    # Ensure TLS 1.2 for downloads
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }

    $gallery = Get-PSRepository -Name 'PSGallery' -ErrorAction SilentlyContinue
    if (-not $gallery) {
        Write-Host "PSGallery not found; registering."
        Register-PSRepository -Default -InstallationPolicy Trusted
    } elseif ($gallery.InstallationPolicy -ne 'Trusted') {
        try {
            Set-PSRepository -Name 'PSGallery' -InstallationPolicy Trusted -ErrorAction Stop
        } catch {
            Write-Warning "Could not set PSGallery as trusted automatically. You may be prompted during install."
        }
    }

    $requiredModules = @(
        @{ Name = 'PSScriptAnalyzer'; Version = '1.22.0' },
        @{ Name = 'Pester'; Version = '5.6.1' }
    )

    foreach ($module in $requiredModules) {
        $installed = Get-Module -ListAvailable -Name $module.Name | Where-Object { $_.Version -ge [version]$module.Version } | Select-Object -First 1
        if ($installed) {
            Write-Host "$($module.Name) $($installed.Version) already present."
            continue
        }
        Write-Host "Installing $($module.Name) $($module.Version) (CurrentUser scope)..."
        Install-Module -Name $module.Name -RequiredVersion $module.Version -Scope CurrentUser -AllowClobber -Force
        $post = Get-Module -ListAvailable -Name $module.Name | Where-Object { $_.Version -ge [version]$module.Version } | Select-Object -First 1
        if (-not $post) {
            throw "Failed to install $($module.Name) $($module.Version)."
        }
        Write-Host "$($module.Name) $($module.Version) installed."
    }
}

function Invoke-PoshQCFormat {
    [CmdletBinding()]
    param(
        [string] $Root = (Get-Location).Path,
        [string] $SettingsPath = $script:PssaSettings,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs
    )

    $ErrorActionPreference = 'Stop'

    if (-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
        throw "PSScriptAnalyzer is not installed. Run Install-PoshQCTools first."
    }
    Import-Module PSScriptAnalyzer -ErrorAction Stop

    if (-not (Test-Path $SettingsPath)) {
        throw "Settings not found: $SettingsPath"
    }

    $files = Get-PoshQCFileList -Root $Root -ExcludeDirs $ExcludeDirs
    if (-not $files) {
        Write-Host "No PowerShell files found under $Root"
        return
    }

    foreach ($file in $files) {
        $original = Get-Content -Raw -Path $file.FullName
        $normalized = $original -replace "`r?`n", "`n"
        $formatted = Invoke-Formatter -ScriptDefinition $normalized -Settings $SettingsPath
        if ($formatted -ne $normalized) {
            Set-Content -Path $file.FullName -Value $formatted -Encoding UTF8
            Write-Host "Formatted: $($file.FullName)"
        }
    }
}

function Invoke-PoshQCAnalyze {
    [CmdletBinding()]
    param(
        [string] $Root = (Get-Location).Path,
        [string] $SettingsPath = $script:PssaSettings,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs
    )

    $ErrorActionPreference = 'Stop'

    if (-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
        throw "PSScriptAnalyzer is not installed. Run Install-PoshQCTools first."
    }
    Import-Module PSScriptAnalyzer -ErrorAction Stop

    if (-not (Test-Path $SettingsPath)) {
        throw "Settings not found: $SettingsPath"
    }

    $files = Get-PoshQCFileList -Root $Root -ExcludeDirs $ExcludeDirs | Where-Object { $_.Extension -in '.ps1', '.psm1' }
    if (-not $files) {
        Write-Host "No PowerShell files found under $Root"
        return
    }

    $results = @()
    foreach ($file in $files) {
        $results += Invoke-ScriptAnalyzer `
            -Path $file.FullName `
            -Settings $SettingsPath `
            -Severity Error, Warning, Information
    }

    if ($results.Count -gt 0) {
        $results | Format-Table -AutoSize
        throw "PSScriptAnalyzer reported $($results.Count) issue(s)."
    }
    Write-Host "PSScriptAnalyzer passed: no findings under $Root"
}

function Invoke-PoshQCTest {
    [CmdletBinding()]
    param(
        [string] $Root = (Get-Location).Path,
        [string] $SettingsPath = $script:PesterSettings,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs
    )

    $ErrorActionPreference = 'Stop'

    if (-not (Get-Module -ListAvailable -Name Pester)) {
        throw "Pester is not installed. Run Install-PoshQCTools first."
    }
    Import-Module Pester -ErrorAction Stop

    if (-not (Test-Path $SettingsPath)) {
        throw "Settings not found: $SettingsPath"
    }

    $settings = Import-PowerShellDataFile -Path $SettingsPath
    $config = New-PesterConfiguration -Hashtable $settings

    # Resolve paths relative to repo root and honor exclusions.
    if ($config.Run.Path) {
        $config.Run.Path = @($config.Run.Path | ForEach-Object { Join-Path $Root $_ })
    }

    # Ensure result directory exists if configured
    if ($config.TestResult.Enabled -and $config.TestResult.OutputPath) {
        $resultDir = Split-Path -Parent $config.TestResult.OutputPath
        if (-not [string]::IsNullOrWhiteSpace($resultDir)) {
            New-Item -ItemType Directory -Path (Join-Path $Root $resultDir) -Force | Out-Null
        }
        $config.TestResult.OutputPath = Join-Path $Root $config.TestResult.OutputPath
    }

    Invoke-Pester -Configuration $config
}

Export-ModuleMember -Function @(
    'Install-PoshQCTools',
    'Invoke-PoshQCFormat',
    'Invoke-PoshQCAnalyze',
    'Invoke-PoshQCTest'
)
