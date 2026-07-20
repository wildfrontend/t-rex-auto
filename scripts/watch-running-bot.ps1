[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$BotProcessId,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$RuntimeRoot = "D:\DinoMutantBot"
$LogFile = Join-Path $RuntimeRoot "app\logs\watchdog.log"

Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class DinoBotWatchExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

function Write-WatchLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $LogFile -Value "$timestamp | $Message"
}

$Continuous = [uint32]2147483648
$SystemRequired = [uint32]2147483649
$PollSeconds = [Math]::Max(5, $PollSeconds)
$StateResult = [DinoBotWatchExecutionState]::SetThreadExecutionState($SystemRequired)
if ($StateResult -eq 0) {
    throw "Unable to register the system-awake request."
}

try {
    Write-WatchLog "Monitoring Bot PID $BotProcessId; system sleep blocked."
    while (Get-Process -Id $BotProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds $PollSeconds
        if (Get-Process -Id $BotProcessId -ErrorAction SilentlyContinue) {
            Write-WatchLog "Bot PID $BotProcessId is still running."
        }
    }
} finally {
    [void][DinoBotWatchExecutionState]::SetThreadExecutionState($Continuous)
    Write-WatchLog "Bot PID $BotProcessId stopped; system-awake request released."
}
