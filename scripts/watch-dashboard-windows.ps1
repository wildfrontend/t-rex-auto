[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8780
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$PythonExecutable = Join-Path $RuntimeRoot "python\python.exe"
$MainScript = Join-Path $AppRoot "main.py"
$ConfigPath = Join-Path $AppRoot "config.json"
$DashboardUrl = "http://127.0.0.1:$Port"
$Mutex = New-Object System.Threading.Mutex(
    $false,
    "Local\DinoMutantBotDashboard-$Port"
)
$OwnsMutex = $false
$DashboardProcess = $null

function Test-DashboardHealth {
    try {
        $Health = Invoke-RestMethod `
            -Uri "$DashboardUrl/api/health" `
            -TimeoutSec 2
        return $Health.ok -eq $true -and $Health.service -eq "dino-dashboard"
    } catch {
        return $false
    }
}

try {
    try {
        $OwnsMutex = $Mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $OwnsMutex = $true
    }
    if (-not $OwnsMutex) {
        exit 0
    }

    while ($true) {
        if (-not (Test-DashboardHealth)) {
            if ($null -eq $DashboardProcess -or $DashboardProcess.HasExited) {
                $Arguments = @(
                    "`"$MainScript`"",
                    "--config", "`"$ConfigPath`"",
                    "dashboard",
                    "--port", [string]$Port
                )
                $DashboardProcess = Start-Process `
                    -FilePath $PythonExecutable `
                    -ArgumentList $Arguments `
                    -WorkingDirectory $RuntimeRoot `
                    -WindowStyle Hidden `
                    -PassThru
            }
        }
        Start-Sleep -Seconds 5
    }
} finally {
    if ($OwnsMutex) {
        $Mutex.ReleaseMutex()
    }
    $Mutex.Dispose()
}
