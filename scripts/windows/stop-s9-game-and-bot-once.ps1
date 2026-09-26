[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Utf8Encoding = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = $Utf8Encoding
$OutputEncoding = $Utf8Encoding

$RuntimeRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Controller = Join-Path $RuntimeRoot ".agents\skills\control-dino-bot\scripts\dashboard-control.ps1"
$LogDirectory = Join-Path $RuntimeRoot "scheduled-logs"
$LogPath = Join-Path $LogDirectory "stop-s9-game-and-bot.log"

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

function Write-RunLog {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -LiteralPath $LogPath -Value "$Timestamp | $Message" -Encoding UTF8
}

function Invoke-DinoControl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Action,
        [switch]$Confirm
    )
    $Arguments = @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $Controller,
        "-Action", $Action,
        "-Instance", $(if ($Action -eq "status") { "all" } else { "main" })
    )
    if ($Confirm) {
        $Arguments += "-Confirm"
    }
    $Raw = & powershell.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed: $Raw"
    }
    return $Raw | ConvertFrom-Json
}

try {
    Write-RunLog "scheduled shutdown started"
    $Status = Invoke-DinoControl -Action "status"
    if (@($Status.serial_collisions).Count -ne 0) {
        throw "ADB serial collision: $($Status.serial_collisions -join ', ')"
    }
    $S9 = @($Status.instances | Where-Object { $_.id -eq "main" })
    if ($S9.Count -ne 1) {
        throw "S9 instance identity is missing or ambiguous"
    }
    if ($S9[0].serial -ne "127.0.0.1:16384" -or [int]$S9[0].status_port -ne 8765) {
        throw "S9 identity mismatch: serial=$($S9[0].serial), port=$($S9[0].status_port)"
    }
    if ($S9[0].running -ne $true) {
        throw "S9 Bot is not running; protected game control is unavailable"
    }

    $Game = Invoke-DinoControl -Action "stop-game" -Confirm
    Write-RunLog "game stop confirmed: $($Game.response.result)"
    $Bot = Invoke-DinoControl -Action "stop" -Confirm
    Write-RunLog "Bot stop accepted: $($Bot.response | ConvertTo-Json -Compress)"

    $After = Invoke-DinoControl -Action "status"
    $S9After = @($After.instances | Where-Object { $_.id -eq "main" })
    if ($S9After.Count -ne 1 -or $S9After[0].running -eq $true) {
        throw "S9 Bot still reports running after stop"
    }
    Write-RunLog "scheduled shutdown completed"
    exit 0
} catch {
    Write-RunLog "FAILED: $($_.Exception.Message)"
    exit 1
}
