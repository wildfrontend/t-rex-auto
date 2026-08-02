[CmdletBinding()]
param(
    [ValidateSet("hatch", "hatch-full", "hatch-hunt", "hatch-filter-test", "hatch-sort-test", "hatch-parent-test", "hatch-attack-test", "hatch-hp-test")]
    [string]$Feature = "hatch",
    [ValidateSet("runtime", "debug")]
    [string]$Mode = "debug",
    [ValidateRange(0, [int]::MaxValue)]
    [int]$MaxActions = 1,
    [ValidateRange(0, [int]::MaxValue)]
    [int]$MaxCycles = 0,
    [ValidateSet("safe", "fast")]
    [string]$Speed = "safe",
    [ValidateRange(1, 65535)]
    [int]$StatusPort = 8766
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
$MainScript = Join-Path $AppRoot "main.py"
$ConfigPath = Join-Path $AppRoot "config.json"

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Windows runtime is not installed. Run start-hunt.cmd for guided setup."
}
if (-not (Test-Path -LiteralPath $MainScript)) {
    throw "Hatch entrypoint not found: $MainScript"
}
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Hatch configuration not found: $ConfigPath"
}

$ExistingBots = @(
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object {
            $_.CommandLine -match "DinoMutantBot.*main.py" -and
            $_.CommandLine -match " run "
        }
)
if ($ExistingBots.Count -gt 0) {
    $ProcessIds = ($ExistingBots | ForEach-Object { $_.ProcessId }) -join ", "
    throw "Another Bot is already running (PID: $ProcessIds). Stop it before starting Hatch."
}

$RunArguments = @(
    $MainScript,
    "--config", $ConfigPath,
    "run",
    "--feature", $Feature,
    "--mode", $Mode,
    "--max-actions", $MaxActions,
    "--max-cycles", $MaxCycles,
    "--speed", $Speed,
    "--status-port", $StatusPort,
    "--verbose"
)

Write-Host "Dino Mutant Bot - $Feature" -ForegroundColor Cyan
Write-Host "Mode: $Mode | Speed: $Speed | Max actions: $MaxActions | Max cycles: $MaxCycles"
Write-Host "Local Hatch status API: http://127.0.0.1:$StatusPort/status"

Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class DinoHatchExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$Continuous = [uint32]2147483648
$SystemRequired = [uint32]2147483649
$BotExitCode = 1
$StateResult = [DinoHatchExecutionState]::SetThreadExecutionState($SystemRequired)
if ($StateResult -eq 0) {
    Write-Warning "Unable to register the system-awake request."
} else {
    Write-Host "System sleep blocked while Hatch runs; display sleep remains enabled."
}

Push-Location $RuntimeRoot
try {
    & $PythonExecutable $RunArguments
    $BotExitCode = $LASTEXITCODE
} finally {
    Pop-Location
    [void][DinoHatchExecutionState]::SetThreadExecutionState($Continuous)
    Write-Host "System-awake request released."
}
exit $BotExitCode
