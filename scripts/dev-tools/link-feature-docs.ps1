# Updates a GitHub issue body to include links to feature docs (user story, spec, plan).
[CmdletBinding()]
param(
    [string] $IssueNumber,
    [string] $FeatureName
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

function Build-FeatureDocumentationBlock {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [string] $FeatureName
    )

    return @"
## Feature Docs
- [User Story](docs/features/active/$FeatureName/user-story.md)
- [Spec](docs/features/active/$FeatureName/spec.md)
- [Plan](docs/features/active/$FeatureName/plan.md)
"@
}

function Set-OrAppendSection {
    [CmdletBinding(SupportsShouldProcess = $true)]
    [OutputType([string])]
    param(
        [string] $Content,
        [string] $SectionHeading,
        [string] $Replacement
    )
    if (-not $PSCmdlet.ShouldProcess($SectionHeading, 'Update section content')) {
        return $Content
    }
    $pattern = "(?ms)^" + [regex]::Escape($SectionHeading) + "\s*\r?\n.*?(?=^\#\#\s+|\z)"
    $regexOptions = [System.Text.RegularExpressions.RegexOptions]::Multiline -bor [System.Text.RegularExpressions.RegexOptions]::Singleline
    $regex = New-Object -TypeName System.Text.RegularExpressions.Regex -ArgumentList @(
        $pattern,
        $regexOptions
    )
    if ($regex.IsMatch($Content)) {
        return $regex.Replace($Content, $Replacement.TrimEnd())
    }
    if ($Content.Trim().Length -eq 0) {
        return $Replacement.TrimEnd()
    }
    return $Content.TrimEnd() + "`n`n" + $Replacement.TrimEnd()
}

function Invoke-LinkFeatureDocument {
    [CmdletBinding()]
    param(
        [string] $IssueNumberParam,
        [string] $FeatureNameParam
    )

    if ([string]::IsNullOrWhiteSpace($IssueNumberParam)) {
        Write-ScriptError "Issue number is required."
    }

    if ([string]::IsNullOrWhiteSpace($FeatureNameParam)) {
        Write-ScriptError "Feature name is required."
    }

    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        Write-ScriptError "gh CLI not found on PATH. Install gh and authenticate first."
    }

    $issueJson = & gh issue view $IssueNumberParam --json body
    if ($LASTEXITCODE -ne 0 -or -not $issueJson) {
        Write-ScriptError "Unable to fetch issue #$IssueNumberParam. Check the number and gh auth."
    }

    $issue = $issueJson | ConvertFrom-Json
    $body = $issue.body
    if ([string]::IsNullOrWhiteSpace($body)) {
        Write-ScriptError "Issue #$IssueNumberParam has an empty body; aborting to avoid overwriting content."
    }

    # Normalize feature name to both underscore and hyphen variants for paths
    $featurePath = $FeatureNameParam
    $docsBlock = Build-FeatureDocumentationBlock -FeatureName $featurePath

    $newBody = Set-OrAppendSection -Content $body -SectionHeading "## Feature Docs" -Replacement $docsBlock

    $tmp = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), '.md')
    Set-Content -Path $tmp -Value $newBody -Encoding UTF8

    & gh issue edit $IssueNumberParam --body-file $tmp
    $exit = $LASTEXITCODE
    Remove-Item $tmp -ErrorAction SilentlyContinue

    if ($exit -eq 0) {
        Write-Output "Updated issue #$IssueNumberParam with Feature Docs links."
    } else {
        Write-ScriptError "Failed to update issue #$IssueNumberParam."
    }
}

if ($MyInvocation.InvocationName -eq '.') {
    return
}

if ($env:POSHQC_SKIP_SCRIPT_EXECUTION) {
    return
}

Invoke-LinkFeatureDocument -IssueNumberParam $IssueNumber -FeatureNameParam $FeatureName
