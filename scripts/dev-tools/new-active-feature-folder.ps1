# Creates an active feature folder from the template.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $FeatureName,
    [ValidateSet("feature", "refactor", "epic", "bug")]
    [string] $Type = "feature",
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
    [CmdletBinding(SupportsShouldProcess = $true)]
    [OutputType([string])]
    param(
        [string] $Content,
        [string] $Name,
        [string] $Body
    )
    if (-not $PSCmdlet.ShouldProcess($Name, 'Set section content')) {
        return $Content
    }
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

function Write-ScriptError {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string] $Message)
    Write-Error -Message $Message
    exit 1
}

if ([string]::IsNullOrWhiteSpace($FeatureName)) {
    Write-ScriptError 'Aborted: no feature name provided. Use -FeatureName.'
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
    Write-ScriptError "Aborted: '$FeatureName' is invalid. Use kebab/underscore-case letters/numbers (e.g., notes-feature or notes_feature)."
}

$workspace = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$template = Join-Path $workspace "docs/features/templates/$Type"
$target = Join-Path $workspace "docs/features/active/$FeatureName"

if (-not (Test-Path $template)) {
    Write-ScriptError "Template folder not found: $template"
}

if ((Test-Path $target) -and -not $Force) {
    Write-ScriptError "Target exists: $target. Use -Force to overwrite."
}

if (-not (Test-Path $target)) {
    New-Item -ItemType Directory -Path $target | Out-Null
}

if ($Type -eq 'bug') {
    $bugTemplateFiles = @(
        (Join-Path $template 'spec.md'),
        (Join-Path $template 'plan.md')
    )
    foreach ($file in $bugTemplateFiles) {
        if (Test-Path $file) {
            Copy-Item -Path $file -Destination $target -Force
        }
    }
} else {
    Copy-Item $template\* $target -Recurse -Force
}
Write-Output "Created/updated: $target"

$filesToOpen = @()
switch ($Type) {
    'feature' {
        $filesToOpen += (Join-Path $target 'user-story.md')
        $filesToOpen += (Join-Path $target 'spec.md')
        $filesToOpen += (Join-Path $target 'plan.md')
    }
    'refactor' {
        $filesToOpen += (Join-Path $target 'spec.md')
        $filesToOpen += (Join-Path $target 'plan.md')
    }
    'epic' {
        $filesToOpen += (Join-Path $target 'initiative.md')
    }
    'bug' {
        $filesToOpen += (Join-Path $target 'spec.md')
        $filesToOpen += (Join-Path $target 'plan.md')
    }
}

# Seed from a similarly named potential, if present (features/refactors)
$potentialFile = $null
$problem = $null
$behavior = $null
$criteriaRaw = $null
$constraints = $null
$tests = $null
$bugSummary = $null
$bugEnvironment = $null
$bugSteps = $null
$bugExpected = $null
$bugActual = $null
$bugLogs = $null
$bugImpact = $null
$bugCause = $null
$bugValidation = $null
if ($Type -ne 'epic') {
    $normalizedName = $FeatureName -replace '_', '-'
    $potentialDir = Join-Path $workspace 'docs/features/potential'
    $promotedDir = Join-Path $potentialDir 'promoted'
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
        if ($Type -eq 'bug') {
            $bugSummary = Get-Section -Content $potentialContent -Name 'Summary'
            $bugEnvironment = Get-Section -Content $potentialContent -Name 'Environment'
            $bugSteps = Get-Section -Content $potentialContent -Name 'Steps to Reproduce'
            $bugExpected = Get-Section -Content $potentialContent -Name 'Expected Behavior'
            $bugActual = Get-Section -Content $potentialContent -Name 'Actual Behavior'
            $bugLogs = Get-Section -Content $potentialContent -Name 'Logs / Screenshots'
            $bugImpact = Get-Section -Content $potentialContent -Name 'Impact / Severity'
            $bugCause = Get-Section -Content $potentialContent -Name 'Suspected Cause / Notes'
            $bugValidation = Get-Section -Content $potentialContent -Name 'Proposed Fix / Validation Ideas'
        } else {
            $problem = Get-Section -Content $potentialContent -Name 'Problem / Why'
            $behavior = Get-Section -Content $potentialContent -Name 'Proposed Behavior'
            $criteriaRaw = Get-Section -Content $potentialContent -Name 'Acceptance Criteria (early draft)'
            $constraints = Get-Section -Content $potentialContent -Name 'Constraints & Risks'
            $tests = Get-Section -Content $potentialContent -Name 'Test Conditions to Consider'
        }

        if (-not $IssueNumber) {
            $issueMatch = [regex]::Match($potentialContent, '^\s*-\s*Issue\s*:\s*#?(\d+)', [System.Text.RegularExpressions.RegexOptions]::Multiline)
            if ($issueMatch.Success) {
                $IssueNumber = $issueMatch.Groups[1].Value
            }
        }
    }
}

$criteria = if ($criteriaRaw) { Format-Checklist $criteriaRaw } else { '' }
$testsFormatted = if ($tests) { Format-Checklist $tests } else { '' }

# Always attempt to fetch issue metadata (if issue number is known) for headers.
$issueMeta = $null
if ($IssueNumber -and (Get-Command gh -ErrorAction SilentlyContinue)) {
    $json = & gh issue view $IssueNumber --json number, title, url, author, updatedAt
    if ($LASTEXITCODE -eq 0 -and $json) {
        $issueMeta = $json | ConvertFrom-Json
    }
}

$issueField = if ($issueMeta.number) { "#$($issueMeta.number)" } elseif ($IssueNumber) { "#$IssueNumber" } else { "#<id>" }
$ownerField = if ($issueMeta.author.login) { $issueMeta.author.login } else { "name" }
$updatedField = if ($issueMeta.updatedAt) { ([datetime]$issueMeta.updatedAt).ToString('yyyy-MM-dd') } else { "YYYY-MM-DD" }

# Paths to active files
$userStoryPath = $null
$specPath = $null
$planPath = $null
$initiativePath = $null
$potentialCopyPath = $null

switch ($Type) {
    'feature' {
        $userStoryPath = Join-Path $target 'user-story.md'
        $specPath = Join-Path $target 'spec.md'
        $planPath = Join-Path $target 'plan.md'
    }
    'refactor' {
        $specPath = Join-Path $target 'spec.md'
        $planPath = Join-Path $target 'plan.md'
    }
    'epic' {
        $initiativePath = Join-Path $target 'initiative.md'
    }
    'bug' {
        $specPath = Join-Path $target 'spec.md'
        $planPath = Join-Path $target 'plan.md'
        if ($potentialFile) {
            $potentialCopyPath = Join-Path $target $potentialFile.Name
            Copy-Item -Path $potentialFile.FullName -Destination $potentialCopyPath -Force
        }
    }
}

# Helper to replace common header placeholders in a template file
function Set-HeaderPlaceholder {
    [CmdletBinding(SupportsShouldProcess = $true)]
    [OutputType([string])]
    param(
        [string] $Content
    )
    if (-not $PSCmdlet.ShouldProcess('header placeholders', 'Update header values')) {
        return $Content
    }
    $result = $Content
    $namePlaceholders = @('<feature-name>', '<refactor-name>', '<epic-name>', '<name>', '<bug-name>')
    foreach ($ph in $namePlaceholders) {
        $result = $result -replace [regex]::Escape($ph), $FeatureName
    }
    $result = [regex]::Replace($result, '#`?<id>`?', $issueField)
    $result = $result -replace '<#id or TBD>', $issueField
    $result = [regex]::Replace($result, '#<tracking-issue>', $issueField)
    $result = [regex]::Replace($result, '^- Owner:\s+(name|<name>)', "- Owner: $ownerField", 'Multiline')
    $result = [regex]::Replace($result, '^- Date:\s+YYYY-MM-DD', "- Date: $updatedField", 'Multiline')
    $result = [regex]::Replace($result, '^- Last Updated:\s+YYYY-MM-DD', "- Last Updated: $updatedField", 'Multiline')
    return $result
}

# Update docs based on type
if ($Type -eq 'feature') {
    # user-story
    if (Test-Path $userStoryPath) {
        $content = Get-Content -Raw -Path $userStoryPath
        $content = Set-HeaderPlaceholder -Content $content
        if ($problem) {
            $content = Set-Section -Content $content -Name 'Problem / Why' -Body $problem
        }
        if ($criteria) {
            $content = Set-Section -Content $content -Name 'Acceptance Criteria' -Body $criteria
        }
        Set-Content -Path $userStoryPath -Value $content -Encoding UTF8
    }

    # spec
    if (Test-Path $specPath) {
        $content = Get-Content -Raw -Path $specPath
        $content = Set-HeaderPlaceholder -Content $content
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
        Set-Content -Path $specPath -Value $content -Encoding UTF8
    }

    # plan
    if (Test-Path $planPath) {
        $content = Get-Content -Raw -Path $planPath
        $content = Set-HeaderPlaceholder -Content $content
        Set-Content -Path $planPath -Value $content -Encoding UTF8
    }
} elseif ($Type -eq 'refactor') {
    if (Test-Path $specPath) {
        $content = Get-Content -Raw -Path $specPath
        $content = Set-HeaderPlaceholder -Content $content
        if ($problem) {
            $content = Set-Section -Content $content -Name 'Intent & Outcomes' -Body $problem
        }
        if ($behavior) {
            $content = Set-Section -Content $content -Name 'Scope (structural changes)' -Body $behavior
        }
        if ($constraints) {
            $content = Set-Section -Content $content -Name 'Risks & Mitigations' -Body $constraints
        }
        if ($testsFormatted) {
            $content = Set-Section -Content $content -Name 'Seeded Test Conditions (from potential)' -Body $testsFormatted
        }
        Set-Content -Path $specPath -Value $content -Encoding UTF8
    }

    if (Test-Path $planPath) {
        $content = Get-Content -Raw -Path $planPath
        $content = Set-HeaderPlaceholder -Content $content
        Set-Content -Path $planPath -Value $content -Encoding UTF8
    }
} elseif ($Type -eq 'epic') {
    if (Test-Path $initiativePath) {
        $content = Get-Content -Raw -Path $initiativePath
        $content = Set-HeaderPlaceholder -Content $content
        Set-Content -Path $initiativePath -Value $content -Encoding UTF8
    }
} elseif ($Type -eq 'bug') {
    if (Test-Path $specPath) {
        $content = Get-Content -Raw -Path $specPath
        $content = Set-HeaderPlaceholder -Content $content

        $contextParts = @()
        if ($bugSummary) {
            $contextParts += $bugSummary
        }
        if ($bugEnvironment) {
            $contextParts += "Environment:`r`n$bugEnvironment"
        }
        if ($bugImpact) {
            $contextParts += "Impact / Severity:`r`n$bugImpact"
        }
        if ($contextParts) {
            $content = Set-Section -Content $content -Name 'Context' -Body ($contextParts -join "`r`n`r`n")
        }

        $reproParts = @()
        if ($bugSteps) {
            $reproParts += "Steps to Reproduce:`r`n$bugSteps"
        }
        $expectedActual = @()
        if ($bugExpected) {
            $expectedActual += "Expected:`r`n$bugExpected"
        }
        if ($bugActual) {
            $expectedActual += "Actual:`r`n$bugActual"
        }
        if ($expectedActual) {
            $reproParts += ($expectedActual -join "`r`n`r`n")
        }
        if ($bugLogs) {
            $reproParts += "Logs / Screenshots:`r`n$bugLogs"
        }
        if ($reproParts) {
            $content = Set-Section -Content $content -Name 'Repro & Evidence' -Body ($reproParts -join "`r`n`r`n")
        }

        if ($bugCause) {
            $content = Set-Section -Content $content -Name 'Root Cause Analysis' -Body $bugCause
        }

        if ($bugValidation) {
            $content = Set-Section -Content $content -Name 'Proposed Fix' -Body $bugValidation
            $content = Set-Section -Content $content -Name 'Test Strategy' -Body $bugValidation
        }

        Set-Content -Path $specPath -Value $content -Encoding UTF8
    }

    if (Test-Path $planPath) {
        $content = Get-Content -Raw -Path $planPath
        $content = Set-HeaderPlaceholder -Content $content
        Set-Content -Path $planPath -Value $content -Encoding UTF8
    }
}

if ($potentialFile) {
    Write-Output "Seeded docs from potential: $($potentialFile.Name)"
}

if ($potentialCopyPath) {
    $filesToOpen += $potentialCopyPath
}

$codeCmd = Get-Command code -ErrorAction SilentlyContinue
if ($codeCmd) {
    $filesToEdit = $filesToOpen | Where-Object { Test-Path $_ }
    if ($filesToEdit.Count -gt 0) {
        Start-Process code -ArgumentList $filesToEdit
    }
} else {
    Write-Warning "VS Code 'code' command not found. Files to edit:"
    $filesToOpen | ForEach-Object { Write-Output "  $_" }
}


