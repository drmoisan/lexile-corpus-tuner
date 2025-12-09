# Creates a GitHub issue from a potential feature file using gh.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $PotentialPath
)

function Write-ScriptError {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string] $Message
    )
    Write-Error -Message $Message
    exit 1
}

$workspace = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$resolved = $null
try {
    $resolved = (Resolve-Path $PotentialPath -ErrorAction Stop).Path
} catch {
    Write-ScriptError "Potential file not found: $PotentialPath"
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-ScriptError "gh CLI not found on PATH. Install gh and authenticate first."
}

$content = Get-Content -Raw -Path $resolved
if ([string]::IsNullOrWhiteSpace($content)) {
    Write-ScriptError "Potential file is empty: $resolved"
}

$headingMatch = [regex]::Match(
    $content,
    '^\s*#\s+(.+)$',
    [System.Text.RegularExpressions.RegexOptions]::Multiline
)
$featureName = $null
if ($headingMatch.Success) {
    $featureName = $headingMatch.Groups[1].Value.Trim()
    $featureName = $featureName -replace '\(Potential\)', ''
    $featureName = $featureName.Trim()
}
if (-not $featureName) {
    $featureName = (Split-Path $resolved -Leaf) -replace '\.md$', ''
}
$issueTitle = "Feature: $featureName"
$featurePath = ($featureName -replace '\s+', '_') -replace '[^A-Za-z0-9_-]', ''

function Get-Section([string] $name) {
    $escaped = [regex]::Escape($name)
    $pattern = "^##\s+$escaped\s*\r?\n(.*?)(?=^##\s+|\z)"
    $m = [regex]::Match(
        $content,
        $pattern,
        [System.Text.RegularExpressions.RegexOptions]::Singleline -bor [System.Text.RegularExpressions.RegexOptions]::Multiline
    )
    if ($m.Success) { return $m.Groups[1].Value.Trim() }
    return ''
}

$problem = Get-Section 'Problem / Why'
$behavior = Get-Section 'Proposed Behavior'
$criteria = Get-Section 'Acceptance Criteria (early draft)'
$constraints = Get-Section 'Constraints & Risks'
$tests = Get-Section 'Test Conditions to Consider'

if (-not $problem) { $problem = '(not provided in potential file)' }
if (-not $behavior) { $behavior = '(not provided in potential file)' }
if (-not $criteria) { $criteria = '(not provided in potential file)' }
if (-not $constraints) { $constraints = '(not provided in potential file)' }
if (-not $tests) { $tests = '(not provided in potential file)' }

$relativePath = $resolved
if (Test-Path $workspace) {
    $relativePath = [System.IO.Path]::GetRelativePath($workspace, $resolved)
}

$body = @"
## Problem / Why
$problem

## Proposed Behavior
$behavior

## Acceptance Criteria
$criteria

## Constraints & Risks
$constraints

## Test Conditions
$tests

## Source
From: $relativePath
"@

$tmp = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), '.md')
Set-Content -Path $tmp -Value $body -Encoding UTF8

Write-Output "Creating issue: $issueTitle"
$result = & gh issue create --title "$issueTitle" --body-file "$tmp" --label "enhancement"
$exit = $LASTEXITCODE

if ($exit -ne 0) {
    Write-Error $result
    Remove-Item $tmp -ErrorAction SilentlyContinue
    exit $exit
}

Write-Output $result

$issueUrl = $null
$issueNumber = $null

$urlMatch = ($result | Select-String -Pattern 'https?://\S+/issues/(\d+)' -AllMatches)
if ($urlMatch.Matches.Count -gt 0) {
    $issueUrl = $urlMatch.Matches[0].Groups[0].Value
    $issueNumber = $urlMatch.Matches[0].Groups[1].Value
}

$issueData = $null
if ($issueNumber) {
    $json = & gh issue view $issueNumber --json number, title, url, author, updatedAt
    if ($LASTEXITCODE -eq 0 -and $json) {
        $issueData = $json | ConvertFrom-Json
    }
}

# Write metadata back to the potential file (issue number, URL, last updated)
if ($issueNumber -and $issueUrl) {
    $rawLines = (Get-Content -Path $resolved -Raw) -split "`r?`n"
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.AddRange([string[]]$rawLines)

    # Update title with issue number
    if ($lines.Count -gt 0) {
        $lines[0] = "# $featureName (Issue #$issueNumber)"
    }

    $metaEnd = $lines.Count
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\s*##\s+') { $metaEnd = $i; break }
    }

    function Set-LineValue {
        [CmdletBinding(SupportsShouldProcess = $true)]
        param(
            [System.Collections.Generic.List[string]] $arr,
            [string] $label,
            [string] $value,
            [ref] $metaEndRef
        )
        if (-not $PSCmdlet.ShouldProcess($label, 'Update line value')) {
            return
        }
        $pattern = "^- $($label):"
        $found = $false
        for ($j = 0; $j -lt $arr.Count; $j++) {
            if ($arr[$j] -match $pattern) {
                $arr[$j] = "- $($label): $value"
                $found = $true
                break
            }
        }
        if (-not $found) {
            $arr.Insert([int]$metaEndRef.Value, "- $($label): $value")
            $metaEndRef.Value++
        }
    }

    $metaEndRef = [ref] $metaEnd
    Set-LineValue -arr $lines -label 'Issue' -value "#$issueNumber" -metaEndRef $metaEndRef
    Set-LineValue -arr $lines -label 'Issue URL' -value $issueUrl -metaEndRef $metaEndRef
    if ($issueData -and $issueData.updatedAt) {
        $updated = ([datetime]$issueData.updatedAt).ToString('yyyy-MM-dd')
        Set-LineValue -arr $lines -label 'Last Updated' -value $updated -metaEndRef $metaEndRef
    }
    $promotedValue = "Promoted -> docs/features/active/$featurePath/ (Issue #$issueNumber)"
    Set-LineValue -arr $lines -label 'Status' -value $promotedValue -metaEndRef $metaEndRef

    Set-Content -Path $resolved -Value $lines -Encoding UTF8
    Write-Output "Updated potential file with issue metadata: $resolved"
}

$promotedDir = Join-Path $workspace 'docs/features/potential/promoted'
if (-not (Test-Path $promotedDir)) {
    New-Item -ItemType Directory -Path $promotedDir | Out-Null
}
$destPath = Join-Path $promotedDir (Split-Path $resolved -Leaf)
Move-Item -Path $resolved -Destination $destPath -Force
Write-Output "Moved potential file to promoted folder: $destPath"

Remove-Item $tmp -ErrorAction SilentlyContinue
exit $exit


