# Creates an active feature folder from the template.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $FeatureName,
    [switch] $Force,
    [string] $IssueNumber
)

function Format-Checklist {
    param([string] $Text)
    $lines = @()
    foreach ($line in ($Text -split "`r?`n")) {
        $trim = $line.Trim()
        if (-not [string]::IsNullOrWhiteSpace($trim)) {
            if ($trim -match '^\-\s*\[?\s*\]') {
                $lines += $trim
            } elseif ($trim -match '^\-') {
                $lines += $trim
            } else {
                $lines += "- [ ] $trim"
            }
        }
    }
    return ($lines -join "`r`n")
}

function Get-Section {
    param(
        [string] $Content,
        [string] $Name
    )
    $escaped = [regex]::Escape($Name)
    $pattern = "^\s*##\s+$escaped\s*\r?\n(.*?)(?=^\s*##\s+|\z)"
    $match = [regex]::Match(
        $Content,
        $pattern,
        [System.Text.RegularExpressions.RegexOptions]::Singleline -bor [System.Text.RegularExpressions.RegexOptions]::Multiline
    )
    if ($match.Success) {
        return $match.Groups[1].Value.Trim()
    }
    return ''
}

function Set-Section {
    param(
        [string] $Content,
        [string] $Name,
        [string] $Body
    )
    if ([string]::IsNullOrWhiteSpace($Body)) {
        return $Content
    }

    $escaped = [regex]::Escape($Name)
    $pattern = "(^##\s+$escaped\s*\r?\n)(.*?)(?=^\s*##\s+|\z)"
    $options = [System.Text.RegularExpressions.RegexOptions]::Singleline -bor [System.Text.RegularExpressions.RegexOptions]::Multiline
    $replacement = "`$1$Body`r`n`r`n"

    if ([regex]::IsMatch($Content, $pattern, $options)) {
        return [regex]::Replace($Content, $pattern, $replacement, $options)
    }

    return $Content.TrimEnd() + "`r`n`r`n## $Name`r`n$Body`r`n"
}

if ([string]::IsNullOrWhiteSpace($FeatureName)) {
    Write-Host 'Aborted: no feature name provided. Use -FeatureName.'
    exit 1
}

# Normalize empty/sentinel issue values to $null so auto-discovery can run.
if ($PSBoundParameters.ContainsKey('IssueNumber')) {
    if ([string]::IsNullOrWhiteSpace($IssueNumber)) {
        $IssueNumber = $null
    } elseif ($IssueNumber.Trim().ToLowerInvariant() -eq 'auto') {
        $IssueNumber = $null
    }
}

$namePattern = '^[a-z0-9]+([-_][a-z0-9]+)*$'
if ($FeatureName -notmatch $namePattern) {
    Write-Host "Aborted: '$FeatureName' is invalid. Use kebab/underscore-case letters/numbers (e.g., notes-feature or notes_feature)."
    exit 1
}

$workspace = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$template = Join-Path $workspace 'docs/features/templates/feature'
$target = Join-Path $workspace "docs/features/active/$FeatureName"

if (-not (Test-Path $template)) {
    Write-Host "Template folder not found: $template"
    exit 1
}

if ((Test-Path $target) -and -not $Force) {
    Write-Host "Target exists: $target. Use -Force to overwrite."
    exit 1
}

if (-not (Test-Path $target)) {
    New-Item -ItemType Directory -Path $target | Out-Null
}

Copy-Item $template\* $target -Recurse -Force
Write-Host "Created/updated: $target"

$filesToOpen = @()
$filesToOpen += (Join-Path $target 'user-story.md')
$filesToOpen += (Join-Path $target 'spec.md')
$filesToOpen += (Join-Path $target 'plan.md')

# Seed from a similarly named potential feature, if present
$normalizedName = $FeatureName -replace '_', '-'
$potentialDir = Join-Path $workspace 'docs/features/potential'
$promotedDir = Join-Path $potentialDir 'promoted'
$potentialFile = $null
if (Test-Path $potentialDir) {
    $potentialFile = Get-ChildItem $potentialDir -File |
        Where-Object {
            $_.Name -like "*$normalizedName*.md" -and
            $_.Name -notin @('template.md', 'README.md')
        } |
        Sort-Object Name -Descending |
        Select-Object -First 1
}
if (-not $potentialFile -and (Test-Path $promotedDir)) {
    $potentialFile = Get-ChildItem $promotedDir -File |
        Where-Object {
            $_.Name -like "*$normalizedName*.md"
        } |
        Sort-Object Name -Descending |
        Select-Object -First 1
}

if ($potentialFile) {
    $potentialContent = Get-Content -Raw -Path $potentialFile.FullName
    $problem = Get-Section -Content $potentialContent -Name 'Problem / Why'
    $behavior = Get-Section -Content $potentialContent -Name 'Proposed Behavior'
    $criteriaRaw = Get-Section -Content $potentialContent -Name 'Acceptance Criteria (early draft)'
    $constraints = Get-Section -Content $potentialContent -Name 'Constraints & Risks'
    $tests = Get-Section -Content $potentialContent -Name 'Test Conditions to Consider'

    $criteria = if ($criteriaRaw) { Format-Checklist $criteriaRaw } else { '' }
    $testsFormatted = if ($tests) { Format-Checklist $tests } else { '' }

    if (-not $IssueNumber) {
        $issueMatch = [regex]::Match($potentialContent, '^\s*-\s*Issue\s*:\s*#?(\d+)', [System.Text.RegularExpressions.RegexOptions]::Multiline)
        if ($issueMatch.Success) {
            $IssueNumber = $issueMatch.Groups[1].Value
        }
    }
}

# Always attempt to fetch issue metadata (if issue number is known) for headers.
$issueMeta = $null
if ($IssueNumber -and (Get-Command gh -ErrorAction SilentlyContinue)) {
    $json = & gh issue view $IssueNumber --json number,title,url,author,updatedAt
    if ($LASTEXITCODE -eq 0 -and $json) {
        $issueMeta = $json | ConvertFrom-Json
    }
}

$issueField = if ($issueMeta.number) { "#$($issueMeta.number)" } elseif ($IssueNumber) { "#$IssueNumber" } else { "#<id>" }
$ownerField = if ($issueMeta.author.login) { $issueMeta.author.login } else { "name" }
$updatedField = if ($issueMeta.updatedAt) { ([datetime]$issueMeta.updatedAt).ToString('yyyy-MM-dd') } else { "YYYY-MM-DD" }

# Paths to active files
$userStoryPath = Join-Path $target 'user-story.md'
$specPath = Join-Path $target 'spec.md'
$planPath = Join-Path $target 'plan.md'

# Helper to replace common header placeholders in a template file
function Set-HeaderPlaceholders {
    param(
        [string] $Content
    )
    $result = $Content
    $result = $result -replace '<feature-name>', $FeatureName
    $result = [regex]::Replace($result, '#`?<id>`?', $issueField)
    $result = [regex]::Replace($result, '^- Owner:\s+name', "- Owner: $ownerField", 'Multiline')
    $result = [regex]::Replace($result, '^- Last Updated:\s+YYYY-MM-DD', "- Last Updated: $updatedField", 'Multiline')
    return $result
}

# Update user-story from template + potential content
if (Test-Path $userStoryPath) {
    $content = Get-Content -Raw -Path $userStoryPath
    $content = Set-HeaderPlaceholders -Content $content
    if ($problem) {
        $content = Set-Section -Content $content -Name 'Problem / Why' -Body $problem
    }
    if ($criteria) {
        $content = Set-Section -Content $content -Name 'Acceptance Criteria' -Body $criteria
    }
    $content = $content -replace '<feature-name>', $FeatureName
    Set-Content -Path $userStoryPath -Value $content -Encoding UTF8
}

# Update spec from template + potential content
if (Test-Path $specPath) {
    $content = Get-Content -Raw -Path $specPath
    $content = Set-HeaderPlaceholders -Content $content
    if ($problem) {
        $content = Set-Section -Content $content -Name 'Overview' -Body $problem
    }
    if ($behavior) {
        $content = Set-Section -Content $content -Name 'Behavior' -Body $behavior
    }
    if ($constraints) {
        $content = Set-Section -Content $content -Name 'Constraints & Risks' -Body $constraints
    }
    if ($testsFormatted) {
        $content = Set-Section -Content $content -Name 'Seeded Test Conditions (from potential)' -Body $testsFormatted
    }
    $content = $content -replace '<feature-name>', $FeatureName
    Set-Content -Path $specPath -Value $content -Encoding UTF8
}

# Update plan headers from template (no section seeding yet)
if (Test-Path $planPath) {
    $content = Get-Content -Raw -Path $planPath
    $content = Set-HeaderPlaceholders -Content $content
    $content = $content -replace '<feature-name>', $FeatureName
    Set-Content -Path $planPath -Value $content -Encoding UTF8
}

if ($potentialFile) {
    Write-Host "Seeded docs from potential: $($potentialFile.Name)"
}

$codeCmd = Get-Command code -ErrorAction SilentlyContinue
if ($codeCmd) {
    $filesToEdit = $filesToOpen | Where-Object { Test-Path $_ }
    if ($filesToEdit.Count -gt 0) {
        Start-Process code -ArgumentList $filesToEdit
    }
} else {
    Write-Host "VS Code 'code' command not found. Files to edit:"
    $filesToOpen | ForEach-Object { Write-Host "  $_" }
}
