# Creates a GitHub issue from a potential feature file using gh.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string] $PotentialPath,
    [ValidateSet('epic', 'feature', 'refactor', 'bug')]
    [string] $PromotionType = 'feature'
)

function Write-ScriptError {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string] $Message
    )
    Write-Error -Message $Message
    throw $Message
}

function Invoke-GhExe {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $GhArgs
    )

    & gh @GhArgs 2>&1
}

function Get-FeatureName {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string] $Content,
        [Parameter(Mandatory = $true)]
        [string] $FilePath
    )
    $headingMatch = [regex]::Match(
        $Content,
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
        $featureName = (Split-Path $FilePath -Leaf) -replace '\.md$', ''
    }
    return $featureName
}

function Get-FeaturePath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string] $FeatureName
    )
    return ($FeatureName -replace '\s+', '_') -replace '[^A-Za-z0-9_-]', ''
}
function Get-Section([string] $name) {
    $escaped = [regex]::Escape($name)
    $pattern = "^##\s+$escaped\s*\r?\n(.*?)(?=^##\s+|\z)"
    $m = [regex]::Match(
        $script:content,
        $pattern,
        [System.Text.RegularExpressions.RegexOptions]::Singleline -bor [System.Text.RegularExpressions.RegexOptions]::Multiline
    )
    if ($m.Success) { return $m.Groups[1].Value.Trim() }
    return ''
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

function Invoke-PotentialToIssue {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string] $PotentialPath,
        [ValidateSet('epic', 'feature', 'refactor', 'bug')]
        [string] $PromotionType = 'feature',
        [Parameter()]
        [ScriptBlock] $GhInvoker,
        [Parameter()]
        [ScriptBlock] $ContentReader,
        [Parameter()]
        [ScriptBlock] $SetContent
    )

    $workspace = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $resolver = $null
    try {
        $resolver = (Resolve-Path $PotentialPath -ErrorAction Stop).Path
    } catch {
        Write-ScriptError "Potential file not found: $PotentialPath"
    }

    $ghInvokerToUse = if ($GhInvoker) { $GhInvoker } else { ${function:Invoke-GhExe} }
    $contentReaderToUse = if ($ContentReader) { $ContentReader } else { { param([string] $Path) Get-Content -Raw -Path $Path } }
    $setContentToUse = if ($SetContent) { $SetContent } else { { param([string]$Path, [object[]]$Value, [string]$Encoding) Set-Content -Path $Path -Value $Value -Encoding $Encoding } }
    if ($ghInvokerToUse -eq ${function:Invoke-GhExe} -and -not (Get-Command gh -ErrorAction SilentlyContinue)) {
        Write-ScriptError "gh CLI not found on PATH. Install gh and authenticate first."
    }

    $script:content = & $contentReaderToUse $resolver
    if ([string]::IsNullOrWhiteSpace($script:content)) {
        Write-ScriptError "Potential file is empty: $resolver"
    }

    $featureName = Get-FeatureName -Content $script:content -FilePath $resolver
    $featurePath = Get-FeaturePath -FeatureName $featureName
    $issueTitle = "Feature: $featureName"

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

    $relativePath = $resolver
    if (Test-Path $workspace) {
        $relativePath = [System.IO.Path]::GetRelativePath($workspace, $resolver)
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
    & $setContentToUse -Path $tmp -Value $body -Encoding UTF8

    Write-Output "Creating issue: $issueTitle (label: $PromotionType)"
    $result = & $ghInvokerToUse -GhArgs @('issue', 'create', '--title', $issueTitle, '--body-file', $tmp, '--label', $PromotionType)
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Error $result
        Remove-Item $tmp -ErrorAction SilentlyContinue
        return $exitCode
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
        $json = & $ghInvokerToUse -GhArgs @('issue', 'view', $issueNumber, '--json', 'number, title, url, author, updatedAt')
        if ($LASTEXITCODE -eq 0 -and $json) {
            $issueData = $json | ConvertFrom-Json
        }
    }

    if ($issueNumber -and $issueUrl) {
        $rawLines = (& $contentReaderToUse $resolver) -split "`r?`n"
        $lines = New-Object System.Collections.Generic.List[string]
        $lines.AddRange([string[]]$rawLines)

        if ($lines.Count -gt 0) {
            $lines[0] = "# $featureName (Issue #$issueNumber)"
        }

        $metaEnd = $lines.Count
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match '^\s*##\s+') { $metaEnd = $i; break }
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

        & $setContentToUse -Path $resolver -Value $lines -Encoding UTF8
        Write-Output "Updated potential file with issue metadata: $resolver"
    }

    $promotedDir = Join-Path $workspace 'docs/features/potential/promoted'
    if (-not (Test-Path $promotedDir)) {
        New-Item -ItemType Directory -Path $promotedDir | Out-Null
    }
    $destPath = Join-Path $promotedDir (Split-Path $resolver -Leaf)
    Move-Item -Path $resolver -Destination $destPath -Force
    Write-Output "Moved potential file to promoted folder: $destPath"

    Remove-Item $tmp -ErrorAction SilentlyContinue
    return $exitCode
}

if ($MyInvocation.InvocationName -ne '.') {
    if (-not $PotentialPath) {
        Write-ScriptError 'PotentialPath is required.'
    }
    $exitValue = Invoke-PotentialToIssue -PotentialPath $PotentialPath -PromotionType $PromotionType
    exit $exitValue
}


