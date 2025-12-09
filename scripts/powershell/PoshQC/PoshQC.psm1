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

<#
.SYNOPSIS
Installs PoshQC dependencies (PSScriptAnalyzer and Pester).
.DESCRIPTION
Ensures PSGallery is trusted and installs required module versions in the CurrentUser scope.
#>
function Install-PoshQCTool {
    [CmdletBinding()]
    param()

    $ErrorActionPreference = 'Stop'

    # Ensure TLS 1.2 for downloads
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    } catch {
        Write-Verbose ("Unable to enforce TLS 1.2 for module install: {0}" -f $_.Exception.Message)
    }

    $gallery = Get-PSRepository -Name 'PSGallery' -ErrorAction SilentlyContinue
    if (-not $gallery) {
        Write-Information "PSGallery not found; registering." -InformationAction Continue
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
            Write-Information "$($module.Name) $($installed.Version) already present." -InformationAction Continue
            continue
        }
        Write-Information "Installing $($module.Name) $($module.Version) (CurrentUser scope)..." -InformationAction Continue
        Install-Module -Name $module.Name -RequiredVersion $module.Version -Scope CurrentUser -AllowClobber -Force
        $post = Get-Module -ListAvailable -Name $module.Name | Where-Object { $_.Version -ge [version]$module.Version } | Select-Object -First 1
        if (-not $post) {
            throw "Failed to install $($module.Name) $($module.Version)."
        }
        Write-Information "$($module.Name) $($module.Version) installed." -InformationAction Continue
    }
}

<#
.SYNOPSIS
Formats PowerShell files with repo PSScriptAnalyzer settings.
.DESCRIPTION
Runs Invoke-Formatter on all PowerShell scripts under the root, excluding configured directories.
#>
function Invoke-PoshQCFormat {
    [CmdletBinding()]
    param(
        [string] $Root = (Get-Location).Path,
        [string] $SettingsPath = $script:PssaSettings,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs
    )

    $ErrorActionPreference = 'Stop'

    if (-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
        throw "PSScriptAnalyzer is not installed. Run Install-PoshQCTool (alias Install-PoshQCTools) first."
    }
    Import-Module PSScriptAnalyzer -ErrorAction Stop

    if (-not (Test-Path $SettingsPath)) {
        throw "Settings not found: $SettingsPath"
    }

    $files = Get-PoshQCFileList -Root $Root -ExcludeDirs $ExcludeDirs
    if (-not $files) {
        Write-Information "No PowerShell files found under $Root" -InformationAction Continue
        return
    }

    foreach ($file in $files) {
        $original = Get-Content -Raw -Path $file.FullName
        $normalized = $original -replace "`r?`n", "`n"
        $formatted = Invoke-Formatter -ScriptDefinition $normalized -Settings $SettingsPath
        if ($formatted -ne $normalized) {
            Set-Content -Path $file.FullName -Value $formatted -Encoding UTF8
            Write-Information "Formatted: $($file.FullName)" -InformationAction Continue
        }
    }
}

<#
.SYNOPSIS
Runs PSScriptAnalyzer with repo settings.
.DESCRIPTION
Analyzes PowerShell scripts under the root and throws if any findings are present.
#>
function Invoke-PoshQCAnalyze {
    [CmdletBinding()]
    param(
        [string] $Root = (Get-Location).Path,
        [string] $SettingsPath = $script:PssaSettings,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs
    )


    $ErrorActionPreference = 'Stop'

    if (-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
        throw "PSScriptAnalyzer is not installed. Run Install-PoshQCTool (alias Install-PoshQCTools) first."
    }
    Import-Module PSScriptAnalyzer -ErrorAction Stop

    if (-not (Test-Path $SettingsPath)) {
        throw "Settings not found: $SettingsPath"
    }

    $files = Get-PoshQCFileList -Root $Root -ExcludeDirs $ExcludeDirs | Where-Object { $_.Extension -in '.ps1', '.psm1' }
    if (-not $files) {
        Write-Information "No PowerShell files found under $Root" -InformationAction Continue
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
    Write-Information "PSScriptAnalyzer passed: no findings under $Root" -InformationAction Continue
}

<#
.SYNOPSIS
Creates a Koverage-friendly copy of coverage output with relative paths.
.DESCRIPTION
Converts Pester CoverageGutters XML by stripping the repository root prefix so tools like Coverage Gutters and Koverage can map files correctly. Accepts file paths or raw XML content and can optionally return the modified content without writing a file.
#>
function Convert-PoshQCCoverageToRelative {
    [CmdletBinding()]
    param(
        [Parameter()][string] $InputPath,
        [Parameter()][string] $OutputPath,
        [Parameter()][string] $RepoRoot = (Get-Location).Path,
        [Parameter()][string] $InputContent,
        [switch] $PassThru
    )

    $ErrorActionPreference = 'Stop'

    if (-not $InputPath -and -not $InputContent) {
        throw 'Either InputPath or InputContent must be provided.'
    }

    $resolvedRoot = (Resolve-Path -Path $RepoRoot -ErrorAction Stop).Path

    $resolvedInputPath = $null
    if ($InputPath) {
        $resolvedInputPath = if ([IO.Path]::IsPathRooted($InputPath)) { $InputPath } else { Join-Path $resolvedRoot $InputPath }
        if (-not (Test-Path $resolvedInputPath)) {
            Write-Information "Coverage file not found; skipping Koverage output: $resolvedInputPath" -InformationAction Continue
            return
        }

        if (-not $InputContent) {
            $InputContent = Get-Content -Path $resolvedInputPath -Raw
        }
    }

    $repoRootClean = [IO.Path]::TrimEndingDirectorySeparator($resolvedRoot)
    # Normalize to forward slashes for consistent regex matching across platforms
    $repoRootNormalized = $repoRootClean -replace '\\', '/'
    $escapedRoot = [regex]::Escape($repoRootNormalized)
    # Replace forward slashes with character class that matches both separators
    $flexiblePattern = $escapedRoot -replace '/', '[\\/]'
    # Match both forward and backslashes after the path
    $escapedPrefixPattern = "$flexiblePattern[\\/]"
    $fixedContent = $InputContent -replace $escapedPrefixPattern, ''

    if ($PassThru) {
        return $fixedContent
    }

    if (-not $OutputPath) {
        $coverageDir = if ($resolvedInputPath) { Split-Path -Parent $resolvedInputPath } else { $resolvedRoot }
        $coverageBase = if ($resolvedInputPath) { [IO.Path]::GetFileNameWithoutExtension($resolvedInputPath) } else { 'powershell-coverage' }
        $OutputPath = Join-Path $coverageDir "$coverageBase.koverage.xml"
    }

    $resolvedOutputPath = if ([IO.Path]::IsPathRooted($OutputPath)) { $OutputPath } else { Join-Path $resolvedRoot $OutputPath }
    $outputDir = Split-Path -Parent $resolvedOutputPath
    if ($outputDir -and -not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }

    Set-Content -Path $resolvedOutputPath -Value $fixedContent -Encoding UTF8
    Write-Information "Wrote Koverage coverage copy: $resolvedOutputPath" -InformationAction Continue
}

<#
.SYNOPSIS
Runs Pester using the repo configuration.
.DESCRIPTION
Builds a Pester configuration from the runsettings file, expands relative paths under the root, and executes tests.
#>
function Invoke-PoshQCTest {
    [CmdletBinding()]
    param(
        [string] $Root = (Get-Location).Path,
        [string] $SettingsPath = $script:PesterSettings,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs,
        [string] $KoverageOutputPath,
        [switch] $DisableKoverageCopy
    )

    $ErrorActionPreference = 'Stop'

    if (-not (Get-Module -ListAvailable -Name Pester)) {
        throw "Pester is not installed. Run Install-PoshQCTool (alias Install-PoshQCTools) first."
    }
    Import-Module Pester -ErrorAction Stop

    if (-not (Test-Path $SettingsPath)) {
        throw "Settings not found: $SettingsPath"
    }

    $settings = Import-PowerShellDataFile -Path $SettingsPath
    $config = New-PesterConfiguration -Hashtable $settings

    # Resolve paths relative to repo root and honor exclusions.
    if ($config.Run.Path.Value) {
        $config.Run.Path = @(
            $config.Run.Path.Value |
                ForEach-Object { Join-Path $Root $_ } |
                    Where-Object { $ExcludeDirs -notcontains (Split-Path -Path $_ -Leaf) }
        )
    }

    if ($ExcludeDirs) {
        $excludedPaths = @($ExcludeDirs | ForEach-Object { Join-Path $Root $_ })
        $existingExclude = if ($config.Run.ExcludePath.Value) { @($config.Run.ExcludePath.Value | ForEach-Object { Join-Path $Root $_ }) } else { @() }
        $config.Run.ExcludePath = $existingExclude + $excludedPaths
    }

    # Ensure result directory exists if configured
    if ($config.TestResult.Enabled.Value -and $config.TestResult.OutputPath.Value) {
        $resultPath = $config.TestResult.OutputPath.Value
        $resultDir = Split-Path -Parent $resultPath
        if (-not [string]::IsNullOrWhiteSpace($resultDir)) {
            New-Item -ItemType Directory -Path (Join-Path $Root $resultDir) -Force | Out-Null
        }
        $config.TestResult.OutputPath = if ([IO.Path]::IsPathRooted($resultPath)) {
            $resultPath
        } else {
            Join-Path $Root $resultPath
        }
    }

    $coverageEnabled = $false
    if ($config.CodeCoverage) {
        if ($config.CodeCoverage.Enabled -is [bool]) {
            $coverageEnabled = $config.CodeCoverage.Enabled
        } elseif ($config.CodeCoverage.Enabled -and $config.CodeCoverage.Enabled.Value) {
            $coverageEnabled = [bool]$config.CodeCoverage.Enabled.Value
        }
    }

    if ($coverageEnabled -and $config.CodeCoverage) {
        if ($config.CodeCoverage.Path.Value) {
            $resolvedCoveragePaths = @(
                $config.CodeCoverage.Path.Value |
                    ForEach-Object {
                        if ([IO.Path]::IsPathRooted($_)) { $_ } else { Join-Path $Root $_ }
                    }
            )
            $config.CodeCoverage.Path = $resolvedCoveragePaths
        }

        if ($config.CodeCoverage.OutputPath.Value) {
            $coveragePath = $config.CodeCoverage.OutputPath.Value
            $coverageDir = Split-Path -Parent $coveragePath
            if (-not [string]::IsNullOrWhiteSpace($coverageDir)) {
                New-Item -ItemType Directory -Path (Join-Path $Root $coverageDir) -Force | Out-Null
            }
            $config.CodeCoverage.OutputPath = if ([IO.Path]::IsPathRooted($coveragePath)) {
                $coveragePath
            } else {
                Join-Path $Root $coveragePath
            }
        }
    }

    $coverageOutputPath = $null
    if ($config.CodeCoverage) {
        if ($config.CodeCoverage.OutputPath -is [string]) {
            $coverageOutputPath = $config.CodeCoverage.OutputPath
        } elseif ($config.CodeCoverage.OutputPath -and $config.CodeCoverage.OutputPath.Value) {
            $coverageOutputPath = $config.CodeCoverage.OutputPath.Value
        }
    }

    $testFiles = @()
    foreach ($path in $config.Run.Path.Value) {
        if (-not (Test-Path $path)) { continue }
        $testFiles += Get-ChildItem -Path $path -Recurse -Include *.Tests.ps1 -File | Where-Object {
            $parts = $_.FullName.Split([IO.Path]::DirectorySeparatorChar, [System.StringSplitOptions]::RemoveEmptyEntries)
            foreach ($dir in $ExcludeDirs) {
                if ($parts -contains $dir) { return $false }
            }
            return $true
        }
    }

    if (-not $testFiles) {
        Write-Information "No Pester test files found under configured paths for root $Root" -InformationAction Continue
        return
    }

    Invoke-Pester -Configuration $config

    $shouldEmitKoverageCopy = -not $DisableKoverageCopy
    if ($shouldEmitKoverageCopy -and $coverageEnabled -and $coverageOutputPath) {
        $derivedKoveragePath = $null
        if ($coverageOutputPath) {
            $coverageBaseName = [IO.Path]::GetFileNameWithoutExtension($coverageOutputPath)
            $coverageParent = Split-Path -Parent $coverageOutputPath
            $derivedKoveragePath = Join-Path $coverageParent "$coverageBaseName.koverage.xml"
        }

        $effectiveKoveragePath = if ($PSBoundParameters.ContainsKey('KoverageOutputPath') -and -not [string]::IsNullOrWhiteSpace($KoverageOutputPath)) {
            $KoverageOutputPath
        } else {
            $derivedKoveragePath
        }

        Convert-PoshQCCoverageToRelative -InputPath $coverageOutputPath -OutputPath $effectiveKoveragePath -RepoRoot $Root
    }
}

Set-Alias -Name Install-PoshQCTools -Value Install-PoshQCTool

Export-ModuleMember -Function @(
    'Install-PoshQCTool',
    'Invoke-PoshQCFormat',
    'Invoke-PoshQCAnalyze',
    'Invoke-PoshQCTest',
    'Convert-PoshQCCoverageToRelative'
) -Alias @('Install-PoshQCTools')
