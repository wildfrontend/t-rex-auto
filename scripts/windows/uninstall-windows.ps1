[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeRoot
)

$ErrorActionPreference = "Stop"
$ResolvedRoot = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd("\")
$PathRoot = [System.IO.Path]::GetPathRoot($ResolvedRoot).TrimEnd("\")
$MainScript = Join-Path $ResolvedRoot "app\main.py"
$DashboardLauncher = Join-Path $ResolvedRoot "start-dashboard.cmd"
$CleanupSource = Join-Path $ResolvedRoot "app\scripts\cleanup-runtime-windows.ps1"
$InstancesPath = Join-Path $ResolvedRoot "instances.json"

if (
    [string]::IsNullOrWhiteSpace($ResolvedRoot) -or
    $ResolvedRoot -eq $PathRoot -or
    -not (Test-Path -LiteralPath $MainScript) -or
    -not (Test-Path -LiteralPath $DashboardLauncher) -or
    -not (Test-Path -LiteralPath $CleanupSource)
) {
    throw "Refusing to uninstall an invalid runtime root: $ResolvedRoot"
}

Write-Host "This will permanently remove:" -ForegroundColor Yellow
Write-Host "  $ResolvedRoot" -ForegroundColor Yellow
$Confirmation = Read-Host "Type DELETE to continue"
if ($Confirmation -cne "DELETE") {
    Write-Host "Uninstall cancelled."
    exit 1
}

function Get-VerifiedLoopbackProcess {
    param(
        [int]$Port,
        [string]$ExpectedService,
        [string]$CommandPattern
    )

    try {
        $HealthPath = if ($ExpectedService -eq "dino-dashboard") {
            "/api/health"
        } else {
            "/health"
        }
        $Health = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$Port$HealthPath" `
            -TimeoutSec 2
        if (
            $Health.ok -ne $true -or
            $Health.service -ne $ExpectedService -or
            $null -eq $Health.process_id
        ) {
            return $null
        }
        $ProcessId = [int]$Health.process_id
        $Connection = Get-NetTCPConnection `
            -LocalPort $Port `
            -State Listen `
            -ErrorAction Stop |
            Where-Object { $_.OwningProcess -eq $ProcessId } |
            Select-Object -First 1
        if ($null -eq $Connection) {
            return $null
        }
        $Process = Get-CimInstance `
            Win32_Process `
            -Filter "ProcessId=$ProcessId" `
            -ErrorAction Stop
        $MainPattern = [regex]::Escape($MainScript)
        if (
            $Process.Name -ine "python.exe" -or
            $Process.CommandLine -notmatch $MainPattern -or
            $Process.CommandLine -notmatch $CommandPattern
        ) {
            return $null
        }
        return $Process
    } catch {
        return $null
    }
}

$BotPorts = @(8765, 8766, 8772, 8773, 8774)
if (Test-Path -LiteralPath $InstancesPath) {
    try {
        $Registry = Get-Content -LiteralPath $InstancesPath -Raw | ConvertFrom-Json
        foreach ($Instance in @($Registry.instances)) {
            $CandidatePort = 0
            if (
                [int]::TryParse([string]$Instance.status_port, [ref]$CandidatePort) -and
                $CandidatePort -ge 1 -and
                $CandidatePort -le 65535 -and
                -not ($BotPorts -contains $CandidatePort)
            ) {
                $BotPorts += $CandidatePort
            }
        }
    } catch {
        Write-Warning "Could not read instances.json; continuing with legacy Bot ports."
    }
}
foreach ($Port in $BotPorts) {
    $PortPattern = "--status-port\s+" + [regex]::Escape([string]$Port) + "(?:\s|$)"
    $Process = Get-VerifiedLoopbackProcess `
        -Port $Port `
        -ExpectedService "dino-mutant-bot-status" `
        -CommandPattern $PortPattern
    if ($null -ne $Process) {
        try {
            Invoke-RestMethod `
                -Method Post `
                -Uri "http://127.0.0.1:$Port/control/stop" `
                -TimeoutSec 3 | Out-Null
            Write-Host "Requested safe Bot stop on port $Port."
        } catch {
            Write-Warning "Bot on port $Port did not accept the safe stop request."
        }
    }
}

$DashboardProcess = Get-VerifiedLoopbackProcess `
    -Port 8780 `
    -ExpectedService "dino-dashboard" `
    -CommandPattern "\sdashboard(?:\s|$)"
if ($null -ne $DashboardProcess) {
    try {
        Invoke-RestMethod `
            -Method Post `
            -Uri "http://127.0.0.1:8780/api/control/shutdown-dashboard" `
            -Headers @{ "X-Dino-Dashboard" = "1" } `
            -TimeoutSec 3 | Out-Null
        Write-Host "Requested Dashboard shutdown."
    } catch {
        Write-Warning "Dashboard did not accept the shutdown request."
    }
}

$Deadline = (Get-Date).AddSeconds(20)
do {
    $Active = $false
    foreach ($Port in $BotPorts) {
        if (
            $null -ne (Get-VerifiedLoopbackProcess `
                -Port $Port `
                -ExpectedService "dino-mutant-bot-status" `
                -CommandPattern "--status-port\s+$Port(?:\s|$)")
        ) {
            $Active = $true
        }
    }
    if (
        $null -ne (Get-VerifiedLoopbackProcess `
            -Port 8780 `
            -ExpectedService "dino-dashboard" `
            -CommandPattern "\sdashboard(?:\s|$)")
    ) {
        $Active = $true
    }
    if (-not $Active) {
        break
    }
    Start-Sleep -Milliseconds 500
} while ((Get-Date) -lt $Deadline)

$AdbExecutable = Join-Path $ResolvedRoot "app\tools\platform-tools\adb.exe"
if (Test-Path -LiteralPath $AdbExecutable) {
    try {
        $AdbCleanup = Start-Process `
            -FilePath $AdbExecutable `
            -ArgumentList @("kill-server") `
            -Wait `
            -PassThru `
            -NoNewWindow
        if ($AdbCleanup.ExitCode -eq 0) {
            Write-Host "Stopped the bundled ADB server."
        }
    } catch {
        Write-Warning "Bundled ADB server cleanup failed; locked files will be removed after reboot."
    }

    $AdbPathPattern = [regex]::Escape($AdbExecutable)
    $BundledAdbProcesses = @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -ieq "adb.exe" -and
                (
                    ([string]$_.ExecutablePath) -ieq $AdbExecutable -or
                    ([string]$_.CommandLine) -match $AdbPathPattern
                )
            }
    )
    foreach ($AdbProcess in $BundledAdbProcesses) {
        try {
            Stop-Process -Id ([int]$AdbProcess.ProcessId) -Force -ErrorAction Stop
            Write-Host "Stopped bundled ADB process PID $($AdbProcess.ProcessId)."
        } catch {
            Write-Warning "Could not stop bundled ADB process PID $($AdbProcess.ProcessId)."
        }
    }

    $AdbDeadline = (Get-Date).AddSeconds(5)
    do {
        $AdbStillRunning = @(
            Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.Name -ieq "adb.exe" -and
                    ([string]$_.ExecutablePath) -ieq $AdbExecutable
                }
        )
        if ($AdbStillRunning.Count -eq 0) {
            break
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $AdbDeadline)
}

$StartupDirectory = [Environment]::GetFolderPath("Startup")
if (-not [string]::IsNullOrWhiteSpace($StartupDirectory)) {
    $LegacyStartupEntry = Join-Path $StartupDirectory "Dino Dashboard Server.cmd"
    Remove-Item -LiteralPath $LegacyStartupEntry -Force -ErrorAction SilentlyContinue
}

$CleanupPath = Join-Path $env:TEMP (
    "dino-runtime-cleanup-" + [guid]::NewGuid().ToString("N") + ".ps1"
)
Copy-Item -LiteralPath $CleanupSource -Destination $CleanupPath -Force
$CleanupArguments = @(
    "-NoLogo",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$CleanupPath`"",
    "-Target", "`"$ResolvedRoot`""
)
Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $CleanupArguments `
    -WorkingDirectory $env:TEMP

Write-Host "Cleanup started. This window will close." -ForegroundColor Green
Write-Host "Any file still locked by Windows is scheduled for deletion after reboot." -ForegroundColor Yellow
exit 0
