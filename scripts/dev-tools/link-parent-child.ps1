# Links a child issue to a parent tracking issue by updating the parent's
# "Child Issues" section and commenting on the child with a parent link.
[CmdletBinding()]
param(
    [string] $ChildIssueNumber,
    [string] $ParentIssueNumber
)

function Write-ScriptError {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string] $Message)
    throw [System.InvalidOperationException]::new($Message)
}

function Invoke-GhCli {
    [CmdletBinding()]
    [OutputType([hashtable])]
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $GhArgs,

        [Parameter()]
        [scriptblock] $InvokeProcess = { param([string[]] $GhArgs) & gh @GhArgs 2>&1 }
    )

    $result = & $InvokeProcess $GhArgs
    $exitCode = $LASTEXITCODE

    return @{ Output = $result; ExitCode = $exitCode }
}

function Test-GhCli {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        Write-ScriptError "gh CLI not found on PATH. Install gh and authenticate first."
    }
}

function Read-IssueNumber {
    param(
        [string] $Label,
        [string] $Value
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        $Value = Read-Host "Enter $Label issue number"
    }
    if ([string]::IsNullOrWhiteSpace($Value)) {
        Write-ScriptError "$Label issue number is required."
    }
    return $Value.Trim()
}

function Get-Issue {
    param(
        [string] $IssueNumber,
        [string] $Label,
        [scriptblock] $InvokeGh = { param([string[]] $GhArgs) Invoke-GhCli -GhArgs $GhArgs }
    )

    $result = & $InvokeGh @('issue', 'view', $IssueNumber, '--json', 'number', 'title', 'url', 'body')
    if ($result.ExitCode -ne 0 -or -not $result.Output) {
        Write-ScriptError "Unable to fetch $Label issue #$IssueNumber. Check the number and gh auth."
    }

    return $result.Output | ConvertFrom-Json
}

function Invoke-LinkParentChild {
    [CmdletBinding()]
    param(
        [string] $ChildIssueNumberParam,
        [string] $ParentIssueNumberParam,
        [scriptblock] $InvokeGh = { param([string[]] $GhArgs) Invoke-GhCli -GhArgs $GhArgs }
    )

    Test-GhCli

    $ChildIssueNumberParam = Read-IssueNumber -Label "child" -Value $ChildIssueNumberParam
    $ParentIssueNumberParam = Read-IssueNumber -Label "parent" -Value $ParentIssueNumberParam

    $childIssue = Get-Issue -IssueNumber $ChildIssueNumberParam -Label "child" -InvokeGh $InvokeGh
    $parentIssue = Get-Issue -IssueNumber $ParentIssueNumberParam -Label "parent" -InvokeGh $InvokeGh

    $parentBody = $parentIssue.body
    if ([string]::IsNullOrWhiteSpace($parentBody)) {
        Write-ScriptError "Parent issue #$ParentIssueNumberParam has an empty body; aborting to avoid overwriting content."
    }

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
        $alreadyListed = $sectionContent -match [regex]::Escape($childIssue.url) -or $sectionContent -match ("#" + [regex]::Escape($childIssue.number))
        if ($alreadyListed) {
            Write-Output "Parent issue already references child #$($childIssue.number); no body update needed."
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
        $response = Read-Host "Parent #$ParentIssueNumberParam has no 'Child Issues' section. Convert to tracking issue and add one? (y/n)"
        if ($response -notin @("y", "Y", "yes", "Yes")) {
            Write-ScriptError "Aborting: parent issue lacks a 'Child Issues' section and conversion was declined."
        }
        $step5Succeeded = $true
        $parentBody = $parentBody.TrimEnd() + "`n`n## Child Issues`n$childEntry`n"
        $parentUpdated = $true
    }

    if (-not $step5Succeeded) {
        Write-ScriptError "Unable to process parent issue body."
    }

    if ($parentUpdated) {
        $tmp = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), '.md')
        Set-Content -Path $tmp -Value $parentBody -Encoding UTF8

        $editResult = & $InvokeGh @('issue', 'edit', $ParentIssueNumberParam, '--body-file', $tmp)
        $editExit = $editResult.ExitCode
        Remove-Item $tmp -ErrorAction SilentlyContinue

        if ($editExit -ne 0) {
            Write-ScriptError "Failed to update parent issue #$ParentIssueNumberParam."
        }
        Write-Output "Updated parent issue #$ParentIssueNumberParam with child link."
    } else {
        Write-Output "No parent body changes were required."
    }

    $childAlreadyLinked = $childIssue.body -match [regex]::Escape($parentIssue.url) -or $childIssue.body -match ("#" + [regex]::Escape($parentIssue.number))
    if ($childAlreadyLinked) {
        Write-Output "Child issue already references parent #$($parentIssue.number); no comment added."
        return
    }

    $comment = "Linked to parent tracking issue #$($parentIssue.number) - $($parentIssue.title)"
    $commentResult = & $InvokeGh @('issue', 'comment', $ChildIssueNumberParam, '--body', $comment)
    if ($commentResult.ExitCode -ne 0) {
        Write-ScriptError "Failed to add parent link comment to child issue #$ChildIssueNumberParam."
    }

    Write-Output "Added parent link comment to child issue #$ChildIssueNumberParam."
}

if ($MyInvocation.InvocationName -eq '.') {
    return
}

if ($env:POSHQC_SKIP_SCRIPT_EXECUTION) {
    return
}

Invoke-LinkParentChild -ChildIssueNumberParam $ChildIssueNumber -ParentIssueNumberParam $ParentIssueNumber
