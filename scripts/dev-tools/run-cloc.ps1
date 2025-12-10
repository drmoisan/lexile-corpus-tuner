param(
    [string]$Path = "$PSScriptRoot/.."
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
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptRoot,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    $toolsRoot = Join-Path -Path $ScriptRoot -ChildPath ".."
    $toolsDir = Join-Path -Path $toolsRoot -ChildPath "tools"

    return @{
        Root       = (Resolve-Path $TargetPath).Path
        ClocExe    = Join-Path -Path $toolsDir -ChildPath "cloc.exe"
        ClocScript = Join-Path -Path $toolsDir -ChildPath "cloc"
    }
}

function Invoke-ClocCount {
    <#
    .SYNOPSIS
        Executes cloc with the appropriate binary for the platform
    #>
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Paths,
        [Parameter(Mandatory = $true)]
        [bool]$IsWindows
    )

    $clocArgs = @("--vcs=git", "--quiet", "--exclude-dir=tools", $Paths.Root)

    if ($IsWindows -and (Test-Path $Paths.ClocExe)) {
        & $Paths.ClocExe @clocArgs
    }
    elseif (Test-Path $Paths.ClocScript) {
        $perl = Get-Command perl -ErrorAction SilentlyContinue
        if (-not $perl) {
            throw "Perl is required to run the bundled cloc script."
        }
        & $perl.Path $Paths.ClocScript @clocArgs
    }
    else {
        throw "Bundled cloc binary not found."
    }
}

# Main execution
Initialize-OutputRendering
$paths = Get-ClocPath -ScriptRoot $PSScriptRoot -TargetPath $Path
$onWindowsPlatform = Test-IsWindows
Invoke-ClocCount -Paths $paths -IsWindows $onWindowsPlatform

