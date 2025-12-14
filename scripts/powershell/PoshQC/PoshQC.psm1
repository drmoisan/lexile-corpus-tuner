$script:ModuleRoot = Split-Path -Parent $PSCommandPath
$script:PssaSettings = Join-Path $ModuleRoot 'settings/pssa.settings.psd1'
$script:PesterSettings = Join-Path $ModuleRoot 'settings/pester.runsettings.psd1'

$script:DefaultExcludedDirs = @(
    '.git', '.venv', 'venv', 'node_modules', 'dist', 'build', '.pytest_cache',
    '__pycache__', '.mypy_cache', '.ruff_cache', '.vscode', '.idea', 'artifacts'
)

<#
.SYNOPSIS
Enumerates PowerShell files in a given root directory.
.DESCRIPTION
Returns all PowerShell (.ps1/.psm1/.psd1) files recursively, excluding specified directories.
#>
function Get-PoshQCFileList {
    [CmdletBinding()]
    [OutputType([System.Object[]])]
    param(
        [Parameter(Mandatory = $true)]
        [string] $Root,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs,
        [scriptblock] $ResolvePath = { param([string] $Path) Resolve-Path -Path $Path -ErrorAction Stop },
        [scriptblock] $EnumerateFiles = { param([string] $Path) Get-ChildItem -Path $Path -Recurse },
        [scriptblock] $ShouldExclude = {
            param($File, [string[]] $ExcludedDirs)
            $parts = $File.FullName -split '[\\/]+' | Where-Object { $_ -ne '' }
            foreach ($dir in $ExcludedDirs) {
                if ($parts -contains $dir) { return $true }
            }
            return $false
        },
        [scriptblock] $IsAllowedExtension = {
            param($File)
            $File.Extension -in '.ps1', '.psm1', '.psd1'
        }
    )

    try {
        $resolvedRoot = & $ResolvePath $Root
        if ($resolvedRoot -isnot [string]) {
            $resolvedRoot = $resolvedRoot.Path
        }
    } catch {
        throw "Failed to resolve root path '$Root': $($_.Exception.Message)"
    }

    $files = @(& $EnumerateFiles $resolvedRoot)
    if (-not $files) { return @() }

    $result = foreach ($file in $files) {
        if (-not (& $IsAllowedExtension $file)) { continue }
        if (& $ShouldExclude $file $ExcludeDirs) { continue }
        $file
    }

    return @($result | Sort-Object -Property FullName -Stable)
}

<#
.SYNOPSIS
Installs PoshQC dependencies (PSScriptAnalyzer and Pester).
.DESCRIPTION
Ensures PSGallery is trusted and installs required module versions in the CurrentUser scope.
#>
function Install-PoshQCTool {
    [CmdletBinding()]
    param(
        [scriptblock] $SetTls = { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 },
        [scriptblock] $GetRepository = { param([string] $Name) Get-PSRepository -Name $Name -ErrorAction SilentlyContinue },
        [scriptblock] $RegisterRepository = { Register-PSRepository -Default -InstallationPolicy Trusted },
        [scriptblock] $SetRepository = { param([string] $Name, [string] $Policy) Set-PSRepository -Name $Name -InstallationPolicy $Policy -ErrorAction Stop },
        [scriptblock] $FindModule = { param([string] $Name) Get-Module -ListAvailable -Name $Name },
        [scriptblock] $InstallModule = { param([string] $Name, [string] $Version) Install-Module -Name $Name -RequiredVersion $Version -Scope CurrentUser -AllowClobber -Force },
        [scriptblock] $Logger = {
            param([string] $Message, [string] $Level = 'Information')
            switch ($Level) {
                'Warning' { Write-Warning $Message }
                'Verbose' { Write-Verbose $Message }
                default { Write-Information $Message -InformationAction Continue }
            }
        }
    )

    $ErrorActionPreference = 'Stop'

    try {
        & $SetTls
    } catch {
        & $Logger "Unable to enforce TLS 1.2 for module install: $($_.Exception.Message)" 'Verbose'
    }
    $gallery = & $GetRepository 'PSGallery'
    if (-not $gallery) {
        & $Logger 'PSGallery not found; registering.'
        & $RegisterRepository
    } elseif ($gallery.InstallationPolicy -ne 'Trusted') {
        try {
            & $SetRepository 'PSGallery' 'Trusted'
        } catch {
            & $Logger 'Could not set PSGallery as trusted automatically. You may be prompted during install.' 'Warning'
        }
    }

    $requiredModules = @(
        @{ Name = 'PSScriptAnalyzer'; Version = '1.22.0' },
        @{ Name = 'Pester'; Version = '5.6.1' }
    )

    foreach ($module in $requiredModules) {
        $installed = & $FindModule $module.Name | Where-Object { $_.Version -ge [version]$module.Version } | Select-Object -First 1
        if ($installed) {
            & $Logger "$($module.Name) $($installed.Version) already present."
            continue
        }

        & $Logger "Installing $($module.Name) $($module.Version) (CurrentUser scope)..."
        try {
            & $InstallModule $module.Name $module.Version
        } catch {
            throw "Failed to install $($module.Name) $($module.Version): $($_.Exception.Message)"
        }

        $post = & $FindModule $module.Name | Where-Object { $_.Version -ge [version]$module.Version } | Select-Object -First 1
        if (-not $post) {
            throw "Failed to install $($module.Name) $($module.Version)."
        }
        & $Logger "$($module.Name) $($module.Version) installed."
    }
}

<#
.SYNOPSIS
Formats PowerShell files using PSScriptAnalyzer.
.DESCRIPTION
Applies PSScriptAnalyzer formatting rules from repo settings to all PowerShell files under Root.
.PARAMETER Root
Root directory to search for PowerShell files. Defaults to current location.
.PARAMETER SettingsPath
Path to PSScriptAnalyzer settings file.
.PARAMETER ExcludeDirs
Directory names to exclude from processing.
#>
function Invoke-PoshQCFormat {
    [CmdletBinding()]
    param(
        [string] $Root = (Get-Location).Path,
        [string] $SettingsPath = $script:PssaSettings,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs,
        [scriptblock] $EnsureModule = {
            param([string] $Name, [string] $ErrorMessage)
            if (-not (Get-Module -ListAvailable -Name $Name)) { throw $ErrorMessage }
            Import-Module $Name -ErrorAction Stop
        },
        [scriptblock] $TestPathExists = { param([string] $Path) Test-Path $Path },
        [scriptblock] $GetFileList = { param([string] $RootPath, [string[]] $Excluded) Get-PoshQCFileList -Root $RootPath -ExcludeDirs $Excluded },
        [scriptblock] $ReadFile = { param([string] $Path) Get-Content -Raw -Path $Path },
        [scriptblock] $WriteFile = { param([string] $Path, [string] $Content) Set-Content -Path $Path -Value $Content -Encoding UTF8 },
        [scriptblock] $FormatContent = { param([string] $Content, [string] $Settings) Invoke-Formatter -ScriptDefinition $Content -Settings $Settings },
        [scriptblock] $Logger = {
            param([string] $Message)
            Write-Information $Message -InformationAction Continue
        }
    )

    $ErrorActionPreference = 'Stop'

    & $EnsureModule 'PSScriptAnalyzer' "PSScriptAnalyzer is not installed. Run Install-PoshQCTool (alias Install-PoshQCTools) first."

    if (-not (& $TestPathExists $SettingsPath)) {
        throw "Settings not found: $SettingsPath"
    }

    $files = @(& $GetFileList $Root $ExcludeDirs)
    if (-not $files) {
        & $Logger "No PowerShell files found under $Root"
        return
    }

    foreach ($file in $files) {
        $original = & $ReadFile $file.FullName
        $normalized = $original -replace "`r?`n", "`n"
        $formatted = & $FormatContent $normalized $SettingsPath
        if ($formatted -ne $normalized) {
            & $WriteFile $file.FullName $formatted
            & $Logger "Formatted: $($file.FullName)"
        } else {
            & $Logger "Already formatted: $($file.FullName)"
        }
    }
}

<#
.SYNOPSIS
Runs PSScriptAnalyzer static analysis on PowerShell files.
.DESCRIPTION
Analyzes all PowerShell files under Root using repo PSScriptAnalyzer settings and reports issues.
.PARAMETER Root
Root directory to search for PowerShell files. Defaults to current location.
.PARAMETER SettingsPath
Path to PSScriptAnalyzer settings file.
.PARAMETER ExcludeDirs
Directory names to exclude from analysis.
#>
function Invoke-PoshQCAnalyze {
    [CmdletBinding()]
    param(
        [string] $Root = (Get-Location).Path,
        [string] $SettingsPath = $script:PssaSettings,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs,
        [scriptblock] $EnsureModule = {
            param([string] $Name, [string] $ErrorMessage)
            if (-not (Get-Module -ListAvailable -Name $Name)) { throw $ErrorMessage }
            Import-Module $Name -ErrorAction Stop
        },
        [scriptblock] $TestPathExists = { param([string] $Path) Test-Path $Path },
        [scriptblock] $GetFileList = { param([string] $RootPath, [string[]] $Excluded) Get-PoshQCFileList -Root $RootPath -ExcludeDirs $Excluded },
        [scriptblock] $AnalyzeFile = {
            param([string] $Path, [string] $Settings)
            Invoke-ScriptAnalyzer -Path $Path -Settings $Settings -Severity Error, Warning, Information -ErrorAction Stop
        },
        [scriptblock] $Logger = {
            param([string] $Message)
            Write-Information $Message -InformationAction Continue
        }
    )

    $ErrorActionPreference = 'Stop'

    & $EnsureModule 'PSScriptAnalyzer' "PSScriptAnalyzer is not installed. Run Install-PoshQCTool (alias Install-PoshQCTools) first."

    if (-not (& $TestPathExists $SettingsPath)) {
        throw "Settings not found: $SettingsPath"
    }

    $files = @(& $GetFileList $Root $ExcludeDirs | Where-Object { $_.Extension -in '.ps1', '.psm1' })
    if (-not $files) {
        & $Logger "No PowerShell files found under $Root"
        return
    }

    $results = @()
    foreach ($file in $files) {
        try {
            $results += & $AnalyzeFile $file.FullName $SettingsPath
        } catch {
            $errorType = $_.Exception.GetType().FullName
            $errorMessage = $_.Exception.Message
            throw "Invoke-ScriptAnalyzer failed for $($file.FullName) ($errorType): $errorMessage"
        }
    }

    if ($results.Count -gt 0) {
        $results | Format-Table -AutoSize
        throw "PSScriptAnalyzer reported $($results.Count) issue(s)."
    }
    & $Logger "PSScriptAnalyzer passed: no findings under $Root"
}

<#
.SYNOPSIS
Converts coverage XML paths from absolute to relative.
.DESCRIPTION
Rewrites Pester coverage XML file paths to be relative to the repo root for Koverage compatibility.
.PARAMETER InputPath
Path to input coverage XML file.
.PARAMETER OutputPath
Path to write relative-path coverage XML.
.PARAMETER RepoRoot
Repository root directory for path relativization.
.PARAMETER InputContent
Alternative to InputPath - provide XML content directly.
.PARAMETER PassThru
Return the converted XML content as output.
#>
function Convert-PoshQCCoverageToRelative {
    [CmdletBinding()]
    param(
        [Parameter()][string] $InputPath,
        [Parameter()][string] $OutputPath,
        [Parameter()][string] $RepoRoot = (Get-Location).Path,
        [Parameter()][string] $InputContent,
        [switch] $PassThru,
        [scriptblock] $ResolvePath = { param([string] $Path) (Resolve-Path -Path $Path -ErrorAction Stop).Path },
        [scriptblock] $JoinPath = { param([string] $Parent, [string] $Child) Join-Path -Path $Parent -ChildPath $Child },
        [scriptblock] $TestPathExists = { param([string] $Path) Test-Path $Path },
        [scriptblock] $ReadContent = { param([string] $Path) Get-Content -Path $Path -Raw },
        [scriptblock] $WriteContent = { param([string] $Path, [string] $Content) Set-Content -Path $Path -Value $Content -Encoding UTF8 },
        [scriptblock] $EnsureDirectory = {
            param([string] $Path)
            $dir = Split-Path -Parent $Path
            if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        },
        [scriptblock] $GetDefaultOutputPath = {
            param([string] $ResolvedInputPath, [string] $ResolvedRoot)
            $coverageDir = if ($ResolvedInputPath) { Split-Path -Parent $ResolvedInputPath } else { $ResolvedRoot }
            $coverageBase = if ($ResolvedInputPath) { [IO.Path]::GetFileNameWithoutExtension($ResolvedInputPath) } else { 'powershell-coverage' }
            Join-Path -Path $coverageDir -ChildPath "$coverageBase.koverage.xml"
        },
        [scriptblock] $Logger = {
            param([string] $Message)
            Write-Information $Message -InformationAction Continue
        }
    )

    $ErrorActionPreference = 'Stop'

    if (-not $InputPath -and -not $InputContent) {
        & $Logger 'No coverage input provided; skipping conversion.'
        return
    }

    $resolvedRoot = $RepoRoot
    try {
        $maybeResolvedRoot = & $ResolvePath $RepoRoot
        if ($maybeResolvedRoot) {
            $resolvedRoot = if ($maybeResolvedRoot -is [string]) { $maybeResolvedRoot } else { $maybeResolvedRoot.Path }
        }
    }
    catch {
        $resolvedRoot = $RepoRoot
    }

    $resolvedInputPath = $null
    if ($InputPath) {
        $resolvedInputPath = if ([IO.Path]::IsPathRooted($InputPath)) { $InputPath } else { & $JoinPath $resolvedRoot $InputPath }
        if (-not (& $TestPathExists $resolvedInputPath)) {
            & $Logger "Coverage file not found; skipping Koverage output: $resolvedInputPath"
            return
        }

        if (-not $InputContent) {
            $InputContent = & $ReadContent $resolvedInputPath
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
        $OutputPath = & $GetDefaultOutputPath $resolvedInputPath $resolvedRoot
    }

    $resolvedOutputPath = if ([IO.Path]::IsPathRooted($OutputPath)) { $OutputPath } else { & $JoinPath $resolvedRoot $OutputPath }
    & $EnsureDirectory $resolvedOutputPath
    & $WriteContent $resolvedOutputPath $fixedContent
    & $Logger "Wrote Koverage coverage copy: $resolvedOutputPath"
}

<#
.SYNOPSIS
Runs Pester tests with coverage reporting.
.DESCRIPTION
Executes Pester tests using repo configuration, generates coverage reports in multiple formats.
.PARAMETER Root
Root directory for test discovery. Defaults to current location.
.PARAMETER SettingsPath
Path to Pester configuration file.
.PARAMETER ExcludeDirs
Directory names to exclude from test/coverage paths.
.PARAMETER KoverageOutputPath
Custom path for Koverage-compatible coverage XML output.
.PARAMETER DisableKoverageCopy
Skip creation of Koverage-friendly coverage copy.
#>
function Invoke-PoshQCTest {
    [CmdletBinding()]
    param(
        [string] $Root = (Get-Location).Path,
        [string] $SettingsPath = $script:PesterSettings,
        [string[]] $ExcludeDirs = $script:DefaultExcludedDirs,
        [string] $KoverageOutputPath,
        [switch] $DisableKoverageCopy,
        [scriptblock] $EnsureModule = {
            param([string] $Name, [string] $ErrorMessage)
            if (-not (Get-Module -ListAvailable -Name $Name)) { throw $ErrorMessage }
            Import-Module $Name -ErrorAction Stop
        },
        [scriptblock] $TestPathExists = { param([string] $Path) Test-Path $Path },
        [scriptblock] $LoadSettings = { param([string] $Path) Import-PowerShellDataFile -Path $Path },
        [scriptblock] $BuildConfiguration = { param($Settings) New-PesterConfiguration -Hashtable $Settings },
        [scriptblock] $ExpandRunPaths = {
            param($Config, [string] $RootPath, [string[]] $Excluded)
            if ($Config.Run.Path.Value) {
                $Config.Run.Path = @(
                    $Config.Run.Path.Value |
                        ForEach-Object { Join-Path $RootPath $_ } |
                            Where-Object { $Excluded -notcontains (Split-Path -Path $_ -Leaf) }
                )
            }

            if ($Excluded) {
                $excludedPaths = @($Excluded | ForEach-Object { Join-Path $RootPath $_ })
                $existingExclude = if ($Config.Run.ExcludePath.Value) { @($Config.Run.ExcludePath.Value | ForEach-Object { Join-Path $RootPath $_ }) } else { @() }
                $Config.Run.ExcludePath = $existingExclude + $excludedPaths
            }

            $Config
        },
        [scriptblock] $EnsureResultPath = {
            param($Config, [string] $RootPath)
            if ($Config.TestResult.Enabled.Value -and $Config.TestResult.OutputPath.Value) {
                $resultPath = $Config.TestResult.OutputPath.Value
                $resultDir = Split-Path -Parent $resultPath
                if (-not [string]::IsNullOrWhiteSpace($resultDir)) {
                    $resolvedResultDir = if ([IO.Path]::IsPathRooted($resultDir)) { $resultDir } else { Join-Path $RootPath $resultDir }
                    New-Item -ItemType Directory -Path $resolvedResultDir -Force | Out-Null
                }
                $Config.TestResult.OutputPath = if ([IO.Path]::IsPathRooted($resultPath)) {
                    $resultPath
                } else {
                    Join-Path $RootPath $resultPath
                }
            }
            $Config
        },
        [scriptblock] $ExpandCoveragePaths = {
            param($Config, [string] $RootPath)
            if (-not $Config.CodeCoverage) { return $Config }

            if ($Config.CodeCoverage.Path.Value) {
                $Config.CodeCoverage.Path = @(
                    $Config.CodeCoverage.Path.Value | ForEach-Object { if ([IO.Path]::IsPathRooted($_)) { $_ } else { Join-Path $RootPath $_ } }
                )
            }

            if ($Config.CodeCoverage.OutputPath.Value) {
                $coveragePath = $Config.CodeCoverage.OutputPath.Value
                $coverageDir = Split-Path -Parent $coveragePath
                if (-not [string]::IsNullOrWhiteSpace($coverageDir)) {
                    $resolvedCoverageDir = if ([IO.Path]::IsPathRooted($coverageDir)) { $coverageDir } else { Join-Path $RootPath $coverageDir }
                    New-Item -ItemType Directory -Path $resolvedCoverageDir -Force | Out-Null
                }
                $Config.CodeCoverage.OutputPath = if ([IO.Path]::IsPathRooted($coveragePath)) {
                    $coveragePath
                } else {
                    Join-Path $RootPath $coveragePath
                }
            }

            $Config
        },
        [scriptblock] $EnumerateTests = {
            param([string[]] $Paths, [string[]] $Excluded, [scriptblock] $TestPathFn)
            $tests = @()
            foreach ($path in $Paths) {
                if (-not (& $TestPathFn $path)) { continue }
                $tests += Get-ChildItem -Path $path -Recurse -Include *.Tests.ps1 | Where-Object {
                    $parts = $_.FullName -split '[\\/]+' | Where-Object { $_ -ne '' }
                    foreach ($dir in $Excluded) {
                        if ($parts -contains $dir) { return $false }
                    }
                    return $true
                }
            }
            @($tests | Sort-Object -Property FullName -Stable)
        },
        [scriptblock] $Logger = {
            param([string] $Message)
            Write-Information $Message -InformationAction Continue
        },
        [scriptblock] $InvokePester = { param($Config) Invoke-Pester -Configuration $Config },
        [scriptblock] $CopyCoverage = {
            param([string] $CoveragePath, [string] $RepoRoot, [string] $KoveragePath)
            Convert-PoshQCCoverageToRelative -InputPath $CoveragePath -OutputPath $KoveragePath -RepoRoot $RepoRoot
        }
    )

    $ErrorActionPreference = 'Stop'

    & $EnsureModule 'Pester' "Pester is not installed. Run Install-PoshQCTool (alias Install-PoshQCTools) first."

    if (-not (& $TestPathExists $SettingsPath)) {
        throw "Settings not found: $SettingsPath"
    }

    $settings = & $LoadSettings $SettingsPath
    $config = & $BuildConfiguration $settings
    $config = & $ExpandRunPaths $config $Root $ExcludeDirs
    $config = & $EnsureResultPath $config $Root
    $config = & $ExpandCoveragePaths $config $Root

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
                $resolvedCoverageDir = if ([IO.Path]::IsPathRooted($coverageDir)) { $coverageDir } else { Join-Path $Root $coverageDir }
                New-Item -ItemType Directory -Path $resolvedCoverageDir -Force | Out-Null
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

    $testFiles = & $EnumerateTests $config.Run.Path.Value $ExcludeDirs $TestPathExists
    if (-not $testFiles) {
        & $Logger "No Pester test files found under configured paths for root $Root"
        return
    }

    & $InvokePester $config

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

        & $CopyCoverage $coverageOutputPath $Root $effectiveKoveragePath
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
