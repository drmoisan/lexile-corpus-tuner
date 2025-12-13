param(
    [string]$Path = "$PSScriptRoot/..",
    [scriptblock]$JoinPath = { param($Parent, $Child) Join-Path -Path $Parent -ChildPath $Child },
    [scriptblock]$ResolvePath = { param($TargetPath) Resolve-Path -Path $TargetPath -ErrorAction Stop },
    [scriptblock]$InvokeProcess = { param($Command, $Arguments) & $Command @($Arguments) }
)

function Initialize-OutputRendering {
    <#
    .SYNOPSIS
        Sets plain text rendering for PowerShell 7+
    #>
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $PSStyle.OutputRendering = 'PlainText'
    }
}

function Test-IsWindows {
    <#
    .SYNOPSIS
        Detects if the current platform is Windows
    #>
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        return $IsWindows
    }
    else {
        return $env:OS -eq "Windows_NT"
    }
}

function Get-ClocPath {
    <#
    .SYNOPSIS
        Constructs paths for cloc executables and target directory
    #>
    [CmdletBinding()]
    [OutputType([System.Collections.Hashtable])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptRoot,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [scriptblock]$JoinPath = { param($Parent, $Child) Join-Path -Path $Parent -ChildPath $Child },
        [scriptblock]$ResolvePath = { param($TargetPath) Resolve-Path -Path $TargetPath -ErrorAction Stop },
        [scriptblock]$TestPath = { param([string]$Path) Test-Path $Path }
    )

    $toolsRoot = & $JoinPath $ScriptRoot ".."
    $toolsDir = & $JoinPath $toolsRoot "tools"

    $targetCandidate = if ([IO.Path]::IsPathRooted($TargetPath)) {
        $TargetPath
    }
    else {
        & $JoinPath $ScriptRoot $TargetPath
    }

    $resolveInputs = @()
    if (-not [IO.Path]::IsPathRooted($TargetPath)) {
        $resolveInputs += $TargetPath
    }
    $resolveInputs += $targetCandidate

    $resolvedTarget = $targetCandidate
    if (& $TestPath $targetCandidate) {
        foreach ($candidate in $resolveInputs) {
            try {
                $resolvedPath = & $ResolvePath $candidate
                if ($resolvedPath) {
                    $resolvedTarget = if ($resolvedPath -is [string]) { $resolvedPath } else { $resolvedPath.Path }
                    break
                }
            }
            catch {
                # Try next candidate - intentionally silent as we iterate through paths
                Write-Verbose "Could not resolve target '$_', trying next candidate"
            }
        }
    }

    return @{
        Root       = $resolvedTarget
        ClocExe    = & $JoinPath $toolsDir "cloc.exe"
        ClocScript = & $JoinPath $toolsDir "cloc"
    }
}

function Invoke-ClocCount {
    <#
    .SYNOPSIS
        Executes cloc with the appropriate binary for the platform
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Paths,
        [Parameter(Mandatory = $true)]
        [Alias('IsWindows')]
        [bool]$IsWindowsPlatform,
        [scriptblock]$TestPath = { param([string]$Path) Test-Path $Path },
        [scriptblock]$FindCommand = { param([string]$Name) Get-Command $Name -ErrorAction SilentlyContinue },
        [scriptblock]$InvokeProcess = { param($Command, $Arguments) & $Command @($Arguments) }
    )

    $clocArgs = @("--vcs=git", "--quiet", "--exclude-dir=tools", $Paths.Root)

    if ($IsWindowsPlatform -and (& $TestPath $Paths.ClocExe)) {
        & $InvokeProcess $Paths.ClocExe $clocArgs
    }
    elseif (& $TestPath $Paths.ClocScript) {
        $perl = & $FindCommand 'perl'
        if (-not $perl) {
            throw "Perl is required to run the bundled cloc script."
        }
        & $InvokeProcess $perl.Path @($Paths.ClocScript) + $clocArgs
    }
    else {
        throw "Bundled cloc binary not found."
    }
}

# Main execution
Initialize-OutputRendering
$paths = Get-ClocPath -ScriptRoot $PSScriptRoot -TargetPath $Path -JoinPath $JoinPath -ResolvePath $ResolvePath
$onWindowsPlatform = Test-IsWindows
Invoke-ClocCount -Paths $paths -IsWindowsPlatform $onWindowsPlatform -InvokeProcess $InvokeProcess

