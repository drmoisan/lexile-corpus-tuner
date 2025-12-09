# Updates a GitHub issue body to include links to feature docs (user story, spec, plan).
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $IssueNumber,
    [Parameter(Mandatory = $true)]
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

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-ScriptError "gh CLI not found on PATH. Install gh and authenticate first."
}

$issueJson = & gh issue view $IssueNumber --json body
if ($LASTEXITCODE -ne 0 -or -not $issueJson) {
    Write-ScriptError "Unable to fetch issue #$IssueNumber. Check the number and gh auth."
}

$issue = $issueJson | ConvertFrom-Json
$body = $issue.body
if ([string]::IsNullOrWhiteSpace($body)) {
    Write-ScriptError "Issue #$IssueNumber has an empty body; aborting to avoid overwriting content."
}

# Normalize feature name to both underscore and hyphen variants for paths
$featurePath = $FeatureName

$docsBlock = @"
## Feature Docs
- [User Story](docs/features/active/$featurePath/user-story.md)
- [Spec](docs/features/active/$featurePath/spec.md)
- [Plan](docs/features/active/$featurePath/plan.md)
"@

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
    $regex = New-Object System.Text.RegularExpressions.Regex(
        $pattern,
        [System.Text.RegularExpressions.RegexOptions]::Multiline -bor [System.Text.RegularExpressions.RegexOptions]::Singleline
    )
    if ($regex.IsMatch($Content)) {
        return $regex.Replace($Content, $Replacement.TrimEnd())
    }
    if ($Content.Trim().Length -eq 0) {
        return $Replacement.TrimEnd()
    }
    return $Content.TrimEnd() + "`n`n" + $Replacement.TrimEnd()
}

$newBody = Set-OrAppendSection -Content $body -SectionHeading "## Feature Docs" -Replacement $docsBlock

$tmp = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), '.md')
Set-Content -Path $tmp -Value $newBody -Encoding UTF8

& gh issue edit $IssueNumber --body-file $tmp
$exit = $LASTEXITCODE
Remove-Item $tmp -ErrorAction SilentlyContinue

if ($exit -eq 0) {
    Write-Output "Updated issue #$IssueNumber with Feature Docs links."
} else {
    Write-ScriptError "Failed to update issue #$IssueNumber."
}
