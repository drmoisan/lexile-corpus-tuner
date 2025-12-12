function global:Import-ScriptFunction {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $resolved = (Resolve-Path -Path $Path).Path
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($resolved, [ref]$null, [ref]$errors)
    if ($errors -and $errors.Count -gt 0) {
        throw "Failed to parse ${resolved}: $($errors[0].Message)"
    }

    $funcAst = $ast.Find(
        {
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq $Name
        },
        $true
    )

    if (-not $funcAst) {
        throw "Function $Name not found in $resolved"
    }

    $functionText = $funcAst.Extent.Text
    $parseErrors = $null
    $parsed = [System.Management.Automation.Language.Parser]::ParseInput($functionText, $resolved, [ref]$null, [ref]$parseErrors)
    if ($parseErrors -and $parseErrors.Count -gt 0) {
        throw "Failed to parse function body for ${resolved}: $($parseErrors[0].Message)"
    }

    return $parsed.GetScriptBlock()
}
