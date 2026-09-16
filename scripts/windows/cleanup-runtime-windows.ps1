[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Target
)

$ErrorActionPreference = "SilentlyContinue"
$ResolvedTarget = [System.IO.Path]::GetFullPath($Target).TrimEnd("\")
$PathRoot = [System.IO.Path]::GetPathRoot($ResolvedTarget).TrimEnd("\")
if ([string]::IsNullOrWhiteSpace($ResolvedTarget) -or $ResolvedTarget -eq $PathRoot) {
    exit 2
}

Start-Sleep -Seconds 3
for ($Attempt = 0; $Attempt -lt 20; $Attempt++) {
    if (-not (Test-Path -LiteralPath $ResolvedTarget)) {
        exit 0
    }
    try {
        Remove-Item -LiteralPath $ResolvedTarget -Recurse -Force -ErrorAction Stop
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not (Test-Path -LiteralPath $ResolvedTarget)) {
    exit 0
}

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class DinoPendingDelete {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern bool MoveFileEx(string existingName, string newName, int flags);
}
"@

$DelayUntilReboot = 4
$Items = @(
    Get-ChildItem -LiteralPath $ResolvedTarget -Force -Recurse -ErrorAction SilentlyContinue |
        Sort-Object { $_.FullName.Length } -Descending
)
foreach ($Item in $Items) {
    [void][DinoPendingDelete]::MoveFileEx(
        $Item.FullName,
        $null,
        $DelayUntilReboot
    )
}
[void][DinoPendingDelete]::MoveFileEx(
    $ResolvedTarget,
    $null,
    $DelayUntilReboot
)
exit 0
