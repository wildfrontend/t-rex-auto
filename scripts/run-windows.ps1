[CmdletBinding()]
param(
    [ValidateSet("runtime", "debug", "training")]
    [string]$Mode = "runtime",
    [int]$MaxActions = 0,
    [int]$MaxCycles = 0,
    [int]$BatchSize = 0,
    [int]$MailAfterHunts = 0,
    [ValidateSet("safe", "fast")]
    [string]$Speed = "fast",
    [int]$ClickDelayMs = -1,
    [int]$DinosaurDelayMs = -1,
    [int]$HuntButtonDelayMs = -1,
    [int]$HuntConfirmDelayMs = -1,
    [int]$IdleDelayMs = -1,
    [ValidateRange(0, 65535)]
    [int]$StatusPort = 8765
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
if (-not (Test-Path $PythonExecutable)) {
    throw "Windows runtime is not installed. Run start-bot.cmd for guided setup."
}

$RunArguments = @(
    (Join-Path $AppRoot "main.py"),
    "--config", (Join-Path $AppRoot "config.json"),
    "run", "--mode", $Mode,
    "--max-actions", $MaxActions,
    "--max-cycles", $MaxCycles,
    "--speed", $Speed,
    "--status-port", $StatusPort,
    "--verbose"
)
if ($BatchSize -gt 0) {
    $RunArguments += @("--batch-size", $BatchSize)
}
if ($MailAfterHunts -gt 0) {
    $RunArguments += @("--mail-after-hunts", $MailAfterHunts)
}
$TimingOverrides = @{
    "--click-delay-ms" = $ClickDelayMs
    "--dinosaur-delay-ms" = $DinosaurDelayMs
    "--hunt-button-delay-ms" = $HuntButtonDelayMs
    "--hunt-confirm-delay-ms" = $HuntConfirmDelayMs
    "--idle-delay-ms" = $IdleDelayMs
}
foreach ($Entry in $TimingOverrides.GetEnumerator()) {
    if ($Entry.Value -ge 0) {
        $RunArguments += @($Entry.Key, $Entry.Value)
    }
}

Write-Host "Bot speed profile: $Speed"
if ($StatusPort -gt 0) {
    Write-Host "Local AI/status API: http://127.0.0.1:$StatusPort/status"
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
