# PowerShell
param(
  [string]$Output = "artifacts/commit_context.txt"
)

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
[Console]::InputEncoding  = $enc
$OutputEncoding           = $enc
$PSDefaultParameterValues['Out-File:Encoding']     = 'utf8'
$PSDefaultParameterValues['Set-Content:Encoding']  = 'utf8'
$PSDefaultParameterValues['Add-Content:Encoding']  = 'utf8'
$PSDefaultParameterValues['Export-Csv:Encoding']   = 'utf8'

function Add-ReportSection {
  param(
    [string]$Title,
    [ScriptBlock]$Cmd,
    [switch]$AllowFail
  )
  Add-Content -Path $script:ReportOutput -Value "`n===== $Title =====`n"
  try {
    if ($Cmd) {
      $result = & $Cmd | Out-String
      Add-Content -Path $script:ReportOutput -Value $result.TrimEnd()
    }
  } catch {
    if ($AllowFail) {
      Add-Content -Path $script:ReportOutput -Value "[n/a]"
    } else {
      throw
    }
  }
}

# Ensure we are inside a Git repo and move to root
git rev-parse --is-inside-work-tree | Out-Null
$root = git rev-parse --show-toplevel
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
Add-ReportSection -Title "Repository remotes" -Cmd { git remote -v }
Add-ReportSection -Title "Current branch" -Cmd { git branch --show-current }
Add-ReportSection -Title "Upstream" -Cmd { git rev-parse --abbrev-ref --symbolic-full-name '@{u}' } -AllowFail
Add-ReportSection -Title "Status (short)" -Cmd { git status -sb }

Add-ReportSection -Title "Staged files (name-status)" -Cmd { git diff --staged --name-status }
Add-ReportSection -Title "Staged diff" -Cmd { git diff --staged }
Add-ReportSection -Title "Unstaged files (name-status)" -Cmd { git diff --name-status }
Add-ReportSection -Title "Unstaged diff" -Cmd { git diff }
Add-ReportSection -Title "Untracked files" -Cmd { git ls-files --others --exclude-standard }

# Summaries
Add-ReportSection -Title "Diff stat (staged + unstaged)" -Cmd { git diff --numstat; git diff --staged --numstat | Sort-Object }
Add-ReportSection -Title "Changed Python files" -Cmd { git diff --name-only HEAD -- '*.py' }

# Baseline context
Add-ReportSection -Title "Last commit (header only)" -Cmd { git show -s --pretty=fuller -1 }

# Placeholder for intent (edit this section in the file if desired)
Add-Content -Path $OutputPath -Value "`n===== Change intent (edit below) =====`n- What/why summary: `n- Breaking changes: `n- Affected modules: `n- Issue/PR refs: `n"

Write-Host "Wrote $OutputPath"
