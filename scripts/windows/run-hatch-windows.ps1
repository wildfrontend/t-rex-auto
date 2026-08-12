[CmdletBinding()]
param(
    [ValidateSet("hatch", "hatch-beginner", "hatch-beginner-hunt", "hatch-full", "hatch-hunt", "hatch-filter-test", "hatch-sort-test", "hatch-parent-test", "hatch-attack-test", "hatch-hp-test", "hatch-stage-hatch", "hatch-stage-attack", "hatch-stage-hp", "hatch-stage-collect", "hatch-stage-cave")]
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
    [int]$StatusPort = 8766,
    [ValidateRange(0, 120)]
    [int]$WaitForExistingSeconds = 0,
    [string]$FailureFile = "",
    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"
trap {
    $FailureMessage = $_.Exception.Message
    if (-not [string]::IsNullOrWhiteSpace($FailureFile)) {
        try {
            $FailureDirectory = Split-Path -Parent $FailureFile
            if (-not [string]::IsNullOrWhiteSpace($FailureDirectory)) {
                New-Item -ItemType Directory -Path $FailureDirectory -Force | Out-Null
            }
            @{
                ok = $false
                error = "runner_failed"
                message = $FailureMessage
                occurred_at = (Get-Date).ToString("o")
            } | ConvertTo-Json -Compress | Set-Content -LiteralPath $FailureFile -Encoding UTF8
        } catch {
            Write-Warning "Unable to write Bot failure detail: $($_.Exception.Message)"
        }
    }
    [Console]::Error.WriteLine("Bot startup failed: $FailureMessage")
    exit 1
}
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
$MainScript = Join-Path $AppRoot "main.py"
$DefaultConfigPath = Join-Path $AppRoot "config.json"
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = $DefaultConfigPath
} else {
    $ConfigPath = [IO.Path]::GetFullPath($ConfigPath)
}

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Windows runtime is not installed. Run start-dashboard.cmd for guided setup."
}
if (-not (Test-Path -LiteralPath $MainScript)) {
    throw "Hatch entrypoint not found: $MainScript"
}
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Hatch configuration not found: $ConfigPath"
}

$ConfigToken = [Regex]::Escape($ConfigPath)
function Get-ExistingBots {
    @(
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object {
            $_.CommandLine -match "main.py" -and
            $_.CommandLine -match " run " -and
            $_.CommandLine -match $ConfigToken
        }
    )
}

$ExistingBots = @(Get-ExistingBots)
if ($ExistingBots.Count -gt 0 -and $WaitForExistingSeconds -gt 0) {
    $ProcessIds = ($ExistingBots | ForEach-Object { $_.ProcessId }) -join ", "
    Write-Host "Waiting up to $WaitForExistingSeconds seconds for previous Bot process(es) to exit (PID: $ProcessIds)."
    $WaitTimer = [Diagnostics.Stopwatch]::StartNew()
    do {
        Start-Sleep -Milliseconds 500
        $ExistingBots = @(Get-ExistingBots)
    } while (
        $ExistingBots.Count -gt 0 -and
        $WaitTimer.Elapsed.TotalSeconds -lt $WaitForExistingSeconds
    )
}
if ($ExistingBots.Count -gt 0) {
    $ProcessIds = ($ExistingBots | ForEach-Object { $_.ProcessId }) -join ", "
    throw "Another Bot instance using this config is already running (PID: $ProcessIds). Stop it before starting Hatch."
}

function Stop-BundledAdbWhenIdle {
    $MainScriptPattern = [regex]::Escape($MainScript)
    $OtherBots = @(
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -match $MainScriptPattern -and
                $_.CommandLine -match " run "
            }
    )
    if ($OtherBots.Count -gt 0) {
        Write-Host "Other Bot instance(s) still running; keeping shared ADB server alive."
        return
    }
    $BundledAdb = Join-Path $AppRoot "tools\platform-tools\adb.exe"
    if (-not (Test-Path -LiteralPath $BundledAdb)) {
        return
    }
    try {
        & $BundledAdb kill-server | Out-Null
    } catch {
        Write-Warning "Bundled ADB kill-server failed: $($_.Exception.Message)"
    }
    $AdbPattern = [regex]::Escape($BundledAdb)
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -ieq "adb.exe" -and
            (
                ([string]$_.ExecutablePath) -ieq $BundledAdb -or
                ([string]$_.CommandLine) -match $AdbPattern
            )
        } |
        ForEach-Object {
            Stop-Process -Id ([int]$_.ProcessId) -Force -ErrorAction SilentlyContinue
        }
    Write-Host "Bundled ADB server stopped because no Bot instances remain."
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

Write-Host "猛龍計畫 - $Feature" -ForegroundColor Cyan
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
    Stop-BundledAdbWhenIdle
    Write-Host "System-awake request released."
}
exit $BotExitCode
