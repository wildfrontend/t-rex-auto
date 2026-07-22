[CmdletBinding()]
param(
    [string]$Speed = "",
    [int]$ClickDelayMs = -1,
    [int]$DinosaurDelayMs = -1,
    [int]$HuntButtonDelayMs = -1,
    [int]$HuntConfirmDelayMs = -1,
    [int]$IdleDelayMs = -1,
    [ValidateRange(1, 65535)]
    [int]$StatusPort = 8765,
    [switch]$SkipDoctor,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$RunnerScript = Join-Path $PSScriptRoot "run-windows.ps1"
$LogRoot = Join-Path $AppRoot "logs"
$Host.UI.RawUI.WindowTitle = "Dino Mutant Bot - Control"

function Read-PositiveTiming {
    param(
        [string]$Label,
        [int]$DefaultValue
    )
    while ($true) {
        $RawValue = Read-Host "$Label in ms [$DefaultValue]"
        if ([string]::IsNullOrWhiteSpace($RawValue)) {
            return $DefaultValue
        }
        $ParsedValue = 0
        if ([int]::TryParse($RawValue, [ref]$ParsedValue) -and $ParsedValue -ge 0) {
            return $ParsedValue
        }
        Write-Host "Please enter a whole number greater than or equal to 0." -ForegroundColor Yellow
    }
}

function Resolve-SpeedSettings {
    param([string]$RequestedSpeed = $Speed)
    $SelectedSpeed = $RequestedSpeed.ToLowerInvariant()
    while ($SelectedSpeed -notin @("fast", "safe", "custom")) {
        Write-Host ""
        Write-Host "Select Bot speed:" -ForegroundColor Cyan
        Write-Host "  1. Fast   (recommended)"
        Write-Host "  2. Safe   (slower transitions)"
        Write-Host "  3. Custom (enter milliseconds)"
        $Choice = Read-Host "Choice [1]"
        if ([string]::IsNullOrWhiteSpace($Choice) -or $Choice -eq "1") {
            $SelectedSpeed = "fast"
        } elseif ($Choice -eq "2") {
            $SelectedSpeed = "safe"
        } elseif ($Choice -eq "3") {
            $SelectedSpeed = "custom"
        } else {
            Write-Host "Invalid choice." -ForegroundColor Yellow
        }
    }

    $Settings = [ordered]@{
        DisplaySpeed = $SelectedSpeed
        Profile = $(if ($SelectedSpeed -eq "safe") { "safe" } else { "fast" })
        ClickDelayMs = $ClickDelayMs
        DinosaurDelayMs = $DinosaurDelayMs
        HuntButtonDelayMs = $HuntButtonDelayMs
        HuntConfirmDelayMs = $HuntConfirmDelayMs
        IdleDelayMs = $IdleDelayMs
    }
    if ($SelectedSpeed -eq "custom") {
        $Settings.ClickDelayMs = Read-PositiveTiming "Default click delay" 1000
        $Settings.DinosaurDelayMs = Read-PositiveTiming "Dinosaur selection delay" 1000
        $Settings.HuntButtonDelayMs = Read-PositiveTiming "Hunt button delay" 3000
        $Settings.HuntConfirmDelayMs = Read-PositiveTiming "Confirm button delay" 2000
        $Settings.IdleDelayMs = Read-PositiveTiming "Idle scan delay" 250
    }
    return [pscustomobject]$Settings
}

function Invoke-EnvironmentCheck {
    $PythonExecutable = Resolve-PythonExecutable
    Write-Host "Checking Python, ADB, assets, and screen capture..." -ForegroundColor Cyan
    $DoctorExitCode = 1
    Push-Location $RuntimeRoot
    try {
        & $PythonExecutable `
            (Join-Path $AppRoot "main.py") `
            "--config" (Join-Path $AppRoot "config.json") `
            "doctor"
        $DoctorExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($DoctorExitCode -ne 0) {
        throw "Environment check failed. Fix the FAIL items above, then run start-bot.cmd again."
    }
    Write-Host "Environment check passed." -ForegroundColor Green
}

function Find-RunningBot {
    try {
        return @(
            Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                Where-Object {
                    $_.CommandLine -match "DinoMutantBot.*main.py" -and
                    $_.CommandLine -match " run "
                }
        )
    } catch {
        Write-Warning "Unable to query existing Bot processes: $($_.Exception.Message)"
        return @()
    }
}

function New-RunnerArguments {
    param([pscustomobject]$Settings)
    $Arguments = @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$RunnerScript`"",
        "-Mode", "runtime",
        "-MaxCycles", "0",
        "-BatchSize", "10",
        "-MailAfterHunts", "30",
        "-Speed", $Settings.Profile,
        "-StatusPort", [string]$StatusPort
    )
    $Overrides = [ordered]@{
        "-ClickDelayMs" = $Settings.ClickDelayMs
        "-DinosaurDelayMs" = $Settings.DinosaurDelayMs
        "-HuntButtonDelayMs" = $Settings.HuntButtonDelayMs
        "-HuntConfirmDelayMs" = $Settings.HuntConfirmDelayMs
        "-IdleDelayMs" = $Settings.IdleDelayMs
    }
    foreach ($Entry in $Overrides.GetEnumerator()) {
        if ($Entry.Value -ge 0) {
            $Arguments += @($Entry.Key, [string]$Entry.Value)
        }
    }
    return $Arguments
}

function Start-BotLogWindow {
    param([pscustomobject]$Settings)
    $Arguments = New-RunnerArguments $Settings
    Write-Host "Starting Bot log window with '$($Settings.DisplaySpeed)' timing..." -ForegroundColor Cyan
    $Process = Start-Process powershell.exe `
        -ArgumentList $Arguments `
        -WorkingDirectory $RuntimeRoot `
        -PassThru
    Start-Sleep -Seconds 1
    if ($Process.HasExited) {
        throw "Bot log window exited immediately with code $($Process.ExitCode)."
    }
    Write-Host "Bot log window PID: $($Process.Id)" -ForegroundColor Green
    return $Process
}

function Stop-BotLogWindow {
    param([System.Diagnostics.Process]$Process)
    if ($null -eq $Process -or $Process.HasExited) {
        return
    }
    Write-Host "Stopping Bot process tree..." -ForegroundColor Yellow
    & taskkill.exe /PID $Process.Id /T /F | Out-Null
    $Process.WaitForExit(5000) | Out-Null
}

function Show-BotStatus {
    param([System.Diagnostics.Process]$Process)
    $ProcessRunning = $null -ne $Process -and -not $Process.HasExited
    $Status = $null
    try {
        $Status = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$StatusPort/status" `
            -TimeoutSec 2
    } catch {
        # The API can be briefly unavailable while the log window starts or stops.
    }
    if (-not $ProcessRunning) {
        Write-Host "Bot status: stopped" -ForegroundColor Red
    } else {
        Write-Host "Bot status: running | stage=$($Status.current_stage) | PID=$($Process.Id)" -ForegroundColor Green
    }
    if ($null -ne $Status) {
        Write-Host "Hunts: $($Status.successful_hunts) | Mailbox cycles: $($Status.mailbox_cycles) | Actions: $($Status.total_actions)"
        Write-Host "Verify failures: $($Status.verification_failures) | Retry exhausted: $($Status.retry_exhausted)"
        Write-Host "Black screens: $($Status.black_screen_detections) | Persisted: $($Status.black_screen_persisted) | Restarts: $($Status.game_restarts)"
        Write-Host "Last hunt: $($Status.last_successful_hunt) | Last log: $($Status.last_log_time)"
        if ($Status.recent_actions.Count -gt 0) {
            Write-Host "--- recent actions ---"
            foreach ($Item in @($Status.recent_actions | Select-Object -Last 5)) {
                Write-Host "$($Item.timestamp) | $($Item.target) | $($Item.action) | $($Item.result)"
            }
            Write-Host "----------------------"
        }
        return
    }
    $LogFile = Join-Path $LogRoot ((Get-Date -Format "yyyyMMdd") + ".log")
    if (Test-Path $LogFile) {
        Write-Host "--- latest log ---"
        Get-Content -LiteralPath $LogFile -Tail 8
        Write-Host "------------------"
    }
}

function Show-AiInterface {
    Write-Host "Read-only localhost API (accessible only on this PC):" -ForegroundColor Cyan
    Write-Host "  Full status : http://127.0.0.1:$StatusPort/status"
    Write-Host "  Actions     : http://127.0.0.1:$StatusPort/actions"
    Write-Host "  Settings    : http://127.0.0.1:$StatusPort/settings"
    Write-Host "  Health      : http://127.0.0.1:$StatusPort/health"
    Write-Host "AI tools can connect to these JSON endpoints without reading the raw log."
}

function Resolve-PythonExecutable {
    $LocalPython = Join-Path $RuntimeRoot "python\python.exe"
    if (Test-Path $LocalPython) {
        return $LocalPython
    }
    $SharedPython = "D:\DinoMutantBot\python\python.exe"
    if (Test-Path $SharedPython) {
        return $SharedPython
    }
    throw "Python runtime was not found."
}

function Invoke-Diagnostics {
    while ($true) {
        Write-Host ""
        Write-Host "Diagnostics:" -ForegroundColor Cyan
        Write-Host "  1. Run environment check"
        Write-Host "  2. Capture a manual ADB screenshot"
        Write-Host "  3. Print full status JSON"
        Write-Host "  4. Show latest 30 raw log lines"
        Write-Host "  5. Open log folder"
        Write-Host "  B. Back"
        $Choice = (Read-Host "Diagnostic command [B]").Trim().ToUpperInvariant()
        if ([string]::IsNullOrWhiteSpace($Choice) -or $Choice -eq "B") {
            return
        } elseif ($Choice -eq "1") {
            Invoke-EnvironmentCheck
        } elseif ($Choice -eq "2") {
            $PythonExecutable = Resolve-PythonExecutable
            $Output = Join-Path $AppRoot ("debug\manual-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".png")
            & $PythonExecutable `
                (Join-Path $AppRoot "main.py") `
                "--config" (Join-Path $AppRoot "config.json") `
                "snapshot" "--backend" "adb" "--output" $Output
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Screenshot saved: $Output" -ForegroundColor Green
            }
        } elseif ($Choice -eq "3") {
            try {
                Invoke-RestMethod `
                    -Uri "http://127.0.0.1:$StatusPort/status" `
                    -TimeoutSec 2 |
                    ConvertTo-Json -Depth 8
            } catch {
                Write-Host "Status API is unavailable: $($_.Exception.Message)" -ForegroundColor Yellow
            }
        } elseif ($Choice -eq "4") {
            $LogFile = Join-Path $LogRoot ((Get-Date -Format "yyyyMMdd") + ".log")
            if (Test-Path $LogFile) {
                Get-Content -LiteralPath $LogFile -Tail 30
            } else {
                Write-Host "No log file exists for today." -ForegroundColor Yellow
            }
        } elseif ($Choice -eq "5") {
            New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
            Start-Process explorer.exe -ArgumentList $LogRoot
        } else {
            Write-Host "Unknown diagnostic command." -ForegroundColor Yellow
        }
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Dino Mutant Bot - Interactive Launcher " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Runtime: $RuntimeRoot"
Write-Host "AI/status API: http://127.0.0.1:$StatusPort/status (local read-only)"

try {
    if (-not $SkipDoctor) {
        Invoke-EnvironmentCheck
    }
    $ExistingBots = Find-RunningBot
    if ($ExistingBots.Count -gt 0) {
        $ProcessIds = ($ExistingBots | ForEach-Object { $_.ProcessId }) -join ", "
        throw "Another Bot is already running (PID: $ProcessIds). Stop it before starting a new one."
    }

    $Settings = Resolve-SpeedSettings -RequestedSpeed $Speed
    Write-Host "Selected speed: $($Settings.DisplaySpeed)" -ForegroundColor Green
    if ($DryRun) {
        Write-Host "Dry run passed. No Bot window was started." -ForegroundColor Green
        exit 0
    }

    $BotProcess = Start-BotLogWindow $Settings
    try {
        while ($true) {
            Write-Host ""
            Write-Host "Control: [S]tatus [T]iming [D]iagnostics [A]I API [R]estart [Q]uit" -ForegroundColor Cyan
            $Command = (Read-Host "Command [S]").Trim().ToUpperInvariant()
            if ([string]::IsNullOrWhiteSpace($Command) -or $Command -eq "S") {
                Show-BotStatus $BotProcess
            } elseif ($Command -eq "T") {
                $NewSettings = Resolve-SpeedSettings -RequestedSpeed ""
                Stop-BotLogWindow $BotProcess
                $Settings = $NewSettings
                $BotProcess = Start-BotLogWindow $Settings
            } elseif ($Command -eq "D") {
                Invoke-Diagnostics
            } elseif ($Command -eq "A") {
                Show-AiInterface
            } elseif ($Command -eq "R") {
                Stop-BotLogWindow $BotProcess
                $BotProcess = Start-BotLogWindow $Settings
            } elseif ($Command -eq "Q") {
                break
            } else {
                Write-Host "Unknown command." -ForegroundColor Yellow
            }
        }
    } finally {
        Stop-BotLogWindow $BotProcess
    }
} catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
