param(
    [string]$Root = "$PSScriptRoot/../..",
    [string[]]$Exclude = @(".git", "node_modules"),
    [switch]$IncludeHidden = $false,
    [switch]$DirectoriesOnly
)

function Show-Tree {
    param(
        [string]$Path,
        [string]$Indent = "",
        [string[]]$ExcludeNames,
        [switch]$IncludeHiddenEntries,
        [switch]$DirectoriesOnly
    )

    $items = Get-ChildItem -LiteralPath $Path -Force | Sort-Object Name | Where-Object {
        $isHidden = ($_.Attributes -band [IO.FileAttributes]::Hidden)
        if (-not $IncludeHiddenEntries -and $isHidden) { return $false }
        if ($ExcludeNames -contains $_.Name) { return $false }
        return $true
    }

    foreach ($item in $items) {
        if ($DirectoriesOnly -and -not $item.PSIsContainer) { continue }

        if ($DirectoriesOnly) {
            Write-Output ("{0}{1}\" -f $Indent, $item.Name)
        }
        else {
            $prefix = if ($item.PSIsContainer) { "[dir]" } else { "     " }
            Write-Output ("{0}{1} {2}" -f $Indent, $prefix, $item.Name)
        }

        if ($item.PSIsContainer) {
            Show-Tree -Path $item.FullName -Indent ($Indent + "    ") -ExcludeNames $ExcludeNames -IncludeHiddenEntries:$IncludeHiddenEntries -DirectoriesOnly:$DirectoriesOnly
        }
    }
}

$rootPath = (Resolve-Path $Root).Path
$repoName = Split-Path -Path $rootPath -Leaf
$modeSuffix = if ($DirectoriesOnly) { " (directories only)" } else { "" }
Write-Output "Tree for $repoName$modeSuffix"
Show-Tree -Path $rootPath -ExcludeNames $Exclude -IncludeHiddenEntries:$IncludeHidden -DirectoriesOnly:$DirectoriesOnly
