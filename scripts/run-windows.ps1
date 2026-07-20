[CmdletBinding()]
param(
    [ValidateSet("runtime", "debug", "training")]
    [string]$Mode = "runtime",
    [int]$MaxActions = 0,
    [int]$MaxCycles = 0,
    [int]$BatchSize = 0,
    [int]$MailAfterHunts = 0
)

$ErrorActionPreference = "Stop"
$RuntimeRoot = "D:\DinoMutantBot"
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
$AppRoot = Join-Path $RuntimeRoot "app"
if (-not (Test-Path $PythonExecutable)) {
    throw "Windows runtime is not installed. Run setup-windows.ps1 first."
}

$RunArguments = @(
    (Join-Path $AppRoot "main.py"),
    "--config", (Join-Path $AppRoot "config.json"),
    "run", "--mode", $Mode,
    "--max-actions", $MaxActions,
    "--max-cycles", $MaxCycles,
    "--verbose"
)
if ($BatchSize -gt 0) {
    $RunArguments += @("--batch-size", $BatchSize)
}
if ($MailAfterHunts -gt 0) {
    $RunArguments += @("--mail-after-hunts", $MailAfterHunts)
}

Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class DinoBotExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$Continuous = [uint32]2147483648
$SystemRequired = [uint32]2147483649
$BotExitCode = 1
$StateResult = [DinoBotExecutionState]::SetThreadExecutionState($SystemRequired)
if ($StateResult -eq 0) {
    Write-Warning "Unable to register the system-awake request."
} else {
    Write-Host "System sleep blocked while Bot runs; display sleep remains enabled."
}

try {
    & $PythonExecutable $RunArguments
    $BotExitCode = $LASTEXITCODE
} finally {
    [void][DinoBotExecutionState]::SetThreadExecutionState($Continuous)
    Write-Host "System-awake request released."
}
exit $BotExitCode
