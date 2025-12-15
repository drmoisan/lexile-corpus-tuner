# PowerShell
[CmdletBinding()]
param(
    [string]$Output = "artifacts/commit_context.txt"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSStyle.OutputRendering = 'PlainText'
}

# Force UTF-8 encoding
if ($PSVersionTable.PSVersion.Major -lt 7) {
    chcp 65001 > $null
}
$enc = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $enc
[Console]::InputEncoding = $enc
$OutputEncoding = $enc
$PSDefaultParameterValues['Out-File:Encoding']    = 'utf8'
$PSDefaultParameterValues['Set-Content:Encoding'] = 'utf8'
$PSDefaultParameterValues['Add-Content:Encoding'] = 'utf8'
$PSDefaultParameterValues['Export-Csv:Encoding']  = 'utf8'

function Invoke-GitExe {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$GitArgs
    )

    # Capture stdout+stderr like the gold standard.
    git @GitArgs 2>&1
}

function Invoke-Git {
    [CmdletBinding()]
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$GitArgs,

        [switch]$AllowNonZeroExit
    )

    $output = Invoke-GitExe -GitArgs $GitArgs
    $code = $LASTEXITCODE

    $lines = @()
    if ($null -ne $output) {
        if ($output -is [System.Array]) { $lines = @($output) }
        else { $lines = @([string]$output) }
    }

    $stdout = ($lines -join "`n")

    if (-not $AllowNonZeroExit -and $code -ne 0) {
        $argLine = ($GitArgs -join ' ')
        throw ("git {0} failed ({1}): {2}" -f $argLine, $code, $stdout)
    }

    return @{ Out = $stdout; Code = $code }
}

function Add-ReportSection {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Title,

        [ScriptBlock]$Cmd,

        [switch]$AllowFail,

        [ScriptBlock]$AddContentFunc = { param($Path, $Value) Add-Content -Path $Path -Value $Value }
    )

    & $AddContentFunc -Path $script:ReportOutput -Value "`n===== $Title =====`n"

    try {
        if ($Cmd) {
            $result = & $Cmd | Out-String
            & $AddContentFunc -Path $script:ReportOutput -Value $result.TrimEnd()
        }
    }
    catch {
        if ($AllowFail) {
            & $AddContentFunc -Path $script:ReportOutput -Value "[n/a]"
        }
        else {
            throw
        }
    }
}

function Invoke-CollectCommitContext {
    [CmdletBinding()]
    param(
        [string]$Output = "artifacts/commit_context.txt"
    )

    # Ensure we are inside a Git repo and move to root
    (Invoke-Git -GitArgs @('rev-parse', '--is-inside-work-tree')).Out | Out-Null
    $root = (Invoke-Git -GitArgs @('rev-parse', '--show-toplevel')).Out
    Set-Location $root

    # Normalize and ensure destination directory exists under repo root.
    $OutputPath = Join-Path -Path $root -ChildPath $Output
    $script:ReportOutput = $OutputPath
    $OutputDir = Split-Path -Parent $OutputPath

    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }

    # Fresh output
    if (Test-Path $OutputPath) { Remove-Item -Force $OutputPath }

    Add-Content -Path $OutputPath -Value "Please generate a commit message based on the following content:`n"

    Add-ReportSection -Title "Repository remotes" -Cmd {
        (Invoke-Git -GitArgs @('remote', '-v')).Out
    }

    Add-ReportSection -Title "Current branch" -Cmd {
        (Invoke-Git -GitArgs @('branch', '--show-current')).Out
    }

    Add-ReportSection -Title "Upstream" -AllowFail -Cmd {
        (Invoke-Git -GitArgs @('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}') -AllowNonZeroExit).Out
    }

    Add-ReportSection -Title "Status (short)" -Cmd {
        (Invoke-Git -GitArgs @('status', '-sb')).Out
    }

    Add-ReportSection -Title "Staged files (name-status)" -Cmd {
        (Invoke-Git -GitArgs @('diff', '--staged', '--name-status')).Out
    }

    Add-ReportSection -Title "Staged diff" -Cmd {
        (Invoke-Git -GitArgs @('diff', '--staged')).Out
    }

    Add-ReportSection -Title "Unstaged files (name-status)" -Cmd {
        (Invoke-Git -GitArgs @('diff', '--name-status')).Out
    }

    Add-ReportSection -Title "Unstaged diff" -Cmd {
        (Invoke-Git -GitArgs @('diff')).Out
    }

    Add-ReportSection -Title "Untracked files" -Cmd {
        (Invoke-Git -GitArgs @('ls-files', '--others', '--exclude-standard')).Out
    }

    # Summaries
    Add-ReportSection -Title "Diff stat (staged + unstaged)" -Cmd {
        @(
            (Invoke-Git -GitArgs @('diff', '--numstat')).Out
            (Invoke-Git -GitArgs @('diff', '--staged', '--numstat')).Out
        ) | Sort-Object
    }

    Add-ReportSection -Title "Changed Python files" -Cmd {
        (Invoke-Git -GitArgs @('diff', '--name-only', 'HEAD', '--', '*.py')).Out
    }

    Add-ReportSection -Title "Last commit (header only)" -Cmd {
        (Invoke-Git -GitArgs @('show', '-s', '--pretty=fuller', '-1')).Out
    }

    Add-Content -Path $OutputPath -Value @"

===== Change intent (edit below) =====
- What/why summary:
- Breaking changes:
- Affected modules:
- Issue/PR refs:

"@

    Write-Output "Wrote $OutputPath"
}

# Do not auto-run when dot-sourced (enables safe function import for tests)
if ($MyInvocation.InvocationName -ne '.') {
    Invoke-CollectCommitContext -Output $Output
}
