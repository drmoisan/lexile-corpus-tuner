# Creates a dated potential feature file from the template and opens it plus backlog.md.
param(
    [string] $ShortName
)

function Test-ValidShortName {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string] $Name
    )
    $shortPattern = '^[a-z0-9]+(-[a-z0-9]+)*$'
    return $Name -cmatch $shortPattern
}

function Get-AuthorName {
    $author = (git config user.name) 2>$null
    if (-not $author -or [string]::IsNullOrWhiteSpace($author)) {
        $author = $env:USERNAME
    }
    if (-not $author) { $author = 'Unknown' }
    return $author
}

function Convert-TemplateContent {
    <#
    .SYNOPSIS
    Replaces placeholders in template content with actual values.

    .DESCRIPTION
    This is a pure string transformation function that does not modify system state.
    It replaces feature-name, date, and author placeholders in the provided content.
    #>
    param(
        [Parameter(Mandatory = $true)]
        [string] $Content,
        [Parameter(Mandatory = $true)]
        [string] $ShortName,
        [Parameter(Mandatory = $true)]
        [string] $Date,
        [Parameter(Mandatory = $true)]
        [string] $Author
    )
    $updatedContent = $Content -replace '<feature-name>', $ShortName
    $updatedContent = $updatedContent -replace 'YYYY-MM-DD', $Date
    $updatedContent = $updatedContent -replace '- Author: name', "- Author: $Author"
    return $updatedContent
}

function Invoke-VSCodeOpen {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Files
    )
    $codeCmd = Get-Command code -ErrorAction SilentlyContinue
    if ($codeCmd) {
        Start-Process code -ArgumentList $Files
        return $true
    }
    return $false
}

# Main script logic
if ([string]::IsNullOrWhiteSpace($ShortName)) {
    Write-Error 'Aborted: no name provided. (Pass -ShortName or use the VS Code task prompt.)'
    exit 1
}

if (-not (Test-ValidShortName -Name $ShortName)) {
    Write-Error "Aborted: '$ShortName' is invalid. Use kebab-case letters/numbers only (e.g., notes-feature)."
    exit 1
}

$workspace = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$today = Get-Date -Format 'yyyy-MM-dd'
$target = Join-Path $workspace "docs/features/potential/$today-$ShortName.md"
$template = Join-Path $workspace 'docs/features/potential/template.md'
$backlog = Join-Path $workspace 'docs/features/backlog.md'

Copy-Item $template $target -Force
Write-Output "Created: $target"

# Populate placeholders in the new file
$author = Get-AuthorName
$content = Get-Content -Raw -Path $target
$content = Convert-TemplateContent -Content $content -ShortName $ShortName -Date $today -Author $author
Set-Content -Path $target -Value $content -Encoding UTF8

$opened = Invoke-VSCodeOpen -Files @($target, $backlog)
if (-not $opened) {
    Write-Warning "VS Code 'code' command not found. Open files manually:"
    Write-Output "  $target"
    Write-Output "  $backlog"
}
