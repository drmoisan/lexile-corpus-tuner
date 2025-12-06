# Links a child issue to a parent tracking issue by updating the parent's
# "Child Issues" section and commenting on the child with a parent link.
[CmdletBinding()]
param(
    [string] $ChildIssueNumber,
    [string] $ParentIssueNumber
)

function Stop-ScriptWithError {
    param([string] $Message)
    Write-Host $Message
    exit 1
}

function Ensure-Gh {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        Stop-ScriptWithError "gh CLI not found on PATH. Install gh and authenticate first."
    }
}

function Prompt-ForIssue {
    param(
        [string] $Label,
        [string] $Value
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        $Value = Read-Host "Enter $Label issue number"
    }
    if ([string]::IsNullOrWhiteSpace($Value)) {
        Stop-ScriptWithError "$Label issue number is required."
    }
    return $Value.Trim()
}

function Get-Issue {
    param([string] $IssueNumber, [string] $Label)
    $json = & gh issue view $IssueNumber --json number,title,url,body
    if ($LASTEXITCODE -ne 0 -or -not $json) {
        Stop-ScriptWithError "Unable to fetch $Label issue #$IssueNumber. Check the number and gh auth."
    }
    return $json | ConvertFrom-Json
}

Ensure-Gh

$ChildIssueNumber = Prompt-ForIssue -Label "child" -Value $ChildIssueNumber
$ParentIssueNumber = Prompt-ForIssue -Label "parent" -Value $ParentIssueNumber

$childIssue = Get-Issue -IssueNumber $ChildIssueNumber -Label "child"
$parentIssue = Get-Issue -IssueNumber $ParentIssueNumber -Label "parent"

$parentBody = $parentIssue.body
if ([string]::IsNullOrWhiteSpace($parentBody)) {
    Stop-ScriptWithError "Parent issue #$ParentIssueNumber has an empty body; aborting to avoid overwriting content."
}

# Match existing tracking patterns: checkbox, #<number>, dash, title (no explicit link needed)
$childEntry = "- [ ] #$($childIssue.number) - $($childIssue.title)"

$headingPattern = "(?ims)^##\s+Child Issues\s*\r?\n(.*?)(?=^\#\#\s+|\z)"
$regexOptions = [System.Text.RegularExpressions.RegexOptions]::Multiline -bor [System.Text.RegularExpressions.RegexOptions]::Singleline
$headingRegex = [System.Text.RegularExpressions.Regex]::new($headingPattern, $regexOptions)

$match = $headingRegex.Match($parentBody)
$parentUpdated = $false
$step5Succeeded = $false

if ($match.Success) {
    $step5Succeeded = $true
    $sectionContent = $match.Groups[1].Value
    $alreadyListed = $sectionContent -match [regex]::Escape($childIssue.url) -or $sectionContent -match ("#"+[regex]::Escape($childIssue.number))
    if ($alreadyListed) {
        Write-Host "Parent issue already references child #$($childIssue.number); no body update needed."
        $parentUpdated = $false
    } else {
        $existingLines = $sectionContent.TrimEnd()
        if ([string]::IsNullOrWhiteSpace($existingLines)) {
            $newSection = "## Child Issues`n$childEntry`n"
        } else {
            $newSection = "## Child Issues`n$existingLines`n$childEntry`n"
        }
        $parentBody = $headingRegex.Replace($parentBody, $newSection.TrimEnd() + "`n")
        $parentUpdated = $true
    }
} else {
    $response = Read-Host "Parent #$ParentIssueNumber has no 'Child Issues' section. Convert to tracking issue and add one? (y/n)"
    if ($response -notin @("y", "Y", "yes", "Yes")) {
        Stop-ScriptWithError "Aborting: parent issue lacks a 'Child Issues' section and conversion was declined."
    }
    $step5Succeeded = $true
    $parentBody = $parentBody.TrimEnd() + "`n`n## Child Issues`n$childEntry`n"
    $parentUpdated = $true
}

if (-not $step5Succeeded) {
    Stop-ScriptWithError "Unable to process parent issue body."
}

if ($parentUpdated) {
    $tmp = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), '.md')
    Set-Content -Path $tmp -Value $parentBody -Encoding UTF8

    & gh issue edit $ParentIssueNumber --body-file $tmp
    $editExit = $LASTEXITCODE
    Remove-Item $tmp -ErrorAction SilentlyContinue

    if ($editExit -ne 0) {
        Stop-ScriptWithError "Failed to update parent issue #$ParentIssueNumber."
    }
    Write-Host "Updated parent issue #$ParentIssueNumber with child link."
} else {
    Write-Host "No parent body changes were required."
}

# Step 6: add link back to parent on the child issue (comment) if not already present.
$childAlreadyLinked = $childIssue.body -match [regex]::Escape($parentIssue.url) -or $childIssue.body -match ("#"+[regex]::Escape($parentIssue.number))
if ($childAlreadyLinked) {
    Write-Host "Child issue already references parent #$($parentIssue.number); no comment added."
    exit 0
}

$comment = "Linked to parent tracking issue #$($parentIssue.number) - $($parentIssue.title)"
& gh issue comment $ChildIssueNumber --body $comment
if ($LASTEXITCODE -ne 0) {
    Stop-ScriptWithError "Failed to add parent link comment to child issue #$ChildIssueNumber."
}

Write-Host "Added parent link comment to child issue #$ChildIssueNumber."
