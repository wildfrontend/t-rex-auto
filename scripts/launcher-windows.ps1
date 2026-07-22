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
$Utf8Encoding = New-Object System.Text.UTF8Encoding $false
[Console]::InputEncoding = $Utf8Encoding
[Console]::OutputEncoding = $Utf8Encoding
$OutputEncoding = $Utf8Encoding
$AppRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Split-Path -Parent $AppRoot
$RunnerScript = Join-Path $PSScriptRoot "run-windows.ps1"
$LogRoot = Join-Path $AppRoot "logs"
$Host.UI.RawUI.WindowTitle = "Dino Mutant Bot - 互動控制台"

function Read-PositiveTiming {
    param(
        [string]$Label,
        [int]$DefaultValue
    )
    while ($true) {
        $RawValue = Read-Host "$Label（毫秒）[$DefaultValue]"
        if ([string]::IsNullOrWhiteSpace($RawValue)) {
            return $DefaultValue
        }
        $ParsedValue = 0
        if ([int]::TryParse($RawValue, [ref]$ParsedValue) -and $ParsedValue -ge 0) {
            return $ParsedValue
        }
        Write-Host "請輸入大於或等於 0 的整數。" -ForegroundColor Yellow
    }
}

function Resolve-SpeedSettings {
    param([string]$RequestedSpeed = $Speed)
    $SelectedSpeed = $RequestedSpeed.ToLowerInvariant()
    while ($SelectedSpeed -notin @("fast", "safe", "custom")) {
        Write-Host ""
        Write-Host "請選擇 Bot 執行速度：" -ForegroundColor Cyan
        Write-Host "  1. 快速（建議）"
        Write-Host "  2. 安全（轉場等待較久）"
        Write-Host "  3. 自訂（手動輸入毫秒）"
        $Choice = Read-Host "請選擇 [1]"
        if ([string]::IsNullOrWhiteSpace($Choice) -or $Choice -eq "1") {
            $SelectedSpeed = "fast"
        } elseif ($Choice -eq "2") {
            $SelectedSpeed = "safe"
        } elseif ($Choice -eq "3") {
            $SelectedSpeed = "custom"
        } else {
            Write-Host "選項無效，請重新輸入。" -ForegroundColor Yellow
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
        $Settings.ClickDelayMs = Read-PositiveTiming "一般點擊等待時間" 1000
        $Settings.DinosaurDelayMs = Read-PositiveTiming "選擇恐龍等待時間" 1000
        $Settings.HuntButtonDelayMs = Read-PositiveTiming "點擊狩獵後等待時間" 3000
        $Settings.HuntConfirmDelayMs = Read-PositiveTiming "確認狩獵後等待時間" 2000
        $Settings.IdleDelayMs = Read-PositiveTiming "無目標時掃描間隔" 250
    }
    return [pscustomobject]$Settings
}

function Invoke-EnvironmentCheck {
    $PythonExecutable = Resolve-PythonExecutable
    Write-Host "正在檢查 Python、ADB、辨識素材與畫面擷取……" -ForegroundColor Cyan
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
        throw "執行環境檢查失敗。請先處理上方 FAIL 項目，再重新執行 start-bot.cmd。"
    }
    Write-Host "執行環境檢查通過。" -ForegroundColor Green
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
        Write-Warning "無法查詢現有 Bot 程序：$($_.Exception.Message)"
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
    $SpeedName = if ($Settings.DisplaySpeed -eq "safe") { "安全" } elseif ($Settings.DisplaySpeed -eq "custom") { "自訂" } else { "快速" }
    Write-Host "正在以「$SpeedName」參數啟動 Bot LOG 視窗……" -ForegroundColor Cyan
    $Process = Start-Process powershell.exe `
        -ArgumentList $Arguments `
        -WorkingDirectory $RuntimeRoot `
        -PassThru
    Start-Sleep -Seconds 1
    if ($Process.HasExited) {
        throw "Bot LOG 視窗立即結束，結束代碼：$($Process.ExitCode)。"
    }
    Write-Host "Bot LOG 視窗已啟動，PID：$($Process.Id)" -ForegroundColor Green
    return $Process
}

function Stop-BotLogWindow {
    param([System.Diagnostics.Process]$Process)
    if ($null -eq $Process -or $Process.HasExited) {
        return
    }
    Write-Host "正在停止 Bot 與相關程序……" -ForegroundColor Yellow
    & taskkill.exe /PID $Process.Id /T /F | Out-Null
    $Process.WaitForExit(5000) | Out-Null
}

function Convert-StageName {
    param([string]$Stage)
    if ([string]::IsNullOrWhiteSpace($Stage)) {
        return "讀取中"
    }
    $Names = @{
        "not_started" = "尚未啟動"
        "stopped" = "已停止"
        "starting" = "啟動中"
        "scanning" = "掃描畫面"
        "planning" = "規劃目標"
        "executing" = "執行操作"
        "verifying" = "驗證結果"
        "retrying" = "重新規劃"
        "waiting_for_picture" = "等待畫面恢復"
        "recovering" = "復原遊戲中"
        "active" = "執行中"
    }
    if ($Names.ContainsKey($Stage)) {
        return $Names[$Stage]
    }
    return $Stage
}

function Convert-TargetName {
    param([string]$Target)
    $Names = @{
        "dinosaur" = "恐龍"
        "hunt_button" = "狩獵按鈕"
        "hunt_confirm_button" = "確認狩獵"
        "hunt_max_group_button" = "最大隊伍"
        "map_exit_nest_button" = "離開巢穴地圖"
        "forest_recenter_button" = "重新進入森林"
        "map_center_egg" = "地圖中央蛋巢"
        "mailbox_button" = "信箱"
        "mail_collect_all_button" = "全部獲取"
        "mail_reward_collect_button" = "領取資源"
        "mail_close_button" = "關閉信箱"
        "duplicate_login_close_button" = "重複登入提示"
        "device_history_confirm_button" = "切回此裝置"
        "startup_growth_result_back" = "自動成長結果"
        "startup_auto_battle_close" = "關閉自動戰鬥"
        "unknown" = "未知目標"
    }
    if ($Names.ContainsKey($Target)) {
        return $Names[$Target]
    }
    return $Target
}

function Convert-ResultName {
    param([string]$Result)
    if ($Result -eq "success") { return "成功" }
    if ($Result -eq "failed") { return "失敗" }
    if ($Result -eq "pending") { return "等待驗證" }
    return $Result
}

function Convert-ActionName {
    param([string]$Action)
    if ($Action -match "^tap") { return $Action -replace "^tap", "點擊" }
    if ($Action -eq "back") { return "返回" }
    if ($Action -match "^swipe") { return $Action -replace "^swipe", "滑動" }
    return $Action
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
        Write-Host "Bot 狀態：已停止" -ForegroundColor Red
    } else {
        $StageName = Convert-StageName $Status.current_stage
        Write-Host "Bot 狀態：執行中｜階段：$StageName｜PID：$($Process.Id)" -ForegroundColor Green
    }
    if ($null -ne $Status) {
        Write-Host "成功狩獵：$($Status.successful_hunts)｜信箱循環：$($Status.mailbox_cycles)｜操作總數：$($Status.total_actions)"
        Write-Host "驗證失敗：$($Status.verification_failures)｜重試耗盡：$($Status.retry_exhausted)"
        Write-Host "偵測黑屏：$($Status.black_screen_detections)｜持續黑屏：$($Status.black_screen_persisted)｜遊戲重啟：$($Status.game_restarts)"
        Write-Host "最近成功狩獵：$($Status.last_successful_hunt)｜最新日誌：$($Status.last_log_time)"
        if ($Status.recent_actions.Count -gt 0) {
            Write-Host "--- 最近執行的操作 ---"
            foreach ($Item in @($Status.recent_actions | Select-Object -Last 5)) {
                $TargetName = Convert-TargetName $Item.target
                $ActionName = Convert-ActionName $Item.action
                $ResultName = Convert-ResultName $Item.result
                Write-Host "$($Item.timestamp)｜$TargetName｜$ActionName｜$ResultName"
            }
            Write-Host "------------------------"
        }
        return
    }
    $LogFile = Join-Path $LogRoot ((Get-Date -Format "yyyyMMdd") + ".log")
    if (Test-Path $LogFile) {
        Write-Host "--- 最新原始日誌 ---"
        Get-Content -LiteralPath $LogFile -Tail 8
        Write-Host "------------------"
    }
}

function Show-AiInterface {
    Write-Host "本機唯讀 AI 接口（只有這台電腦可以存取）：" -ForegroundColor Cyan
    Write-Host "  完整狀態：http://127.0.0.1:$StatusPort/status"
    Write-Host "  最近操作：http://127.0.0.1:$StatusPort/actions"
    Write-Host "  目前設定：http://127.0.0.1:$StatusPort/settings"
    Write-Host "  健康檢查：http://127.0.0.1:$StatusPort/health"
    Write-Host "AI 工具可直接讀取以上 JSON，不需要解析原始 LOG。"
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
    throw "找不到 Python 執行環境。"
}

function Invoke-Diagnostics {
    while ($true) {
        Write-Host ""
        Write-Host "診斷工具：" -ForegroundColor Cyan
        Write-Host "  1. 執行完整環境檢查"
        Write-Host "  2. 手動擷取一張 ADB 畫面"
        Write-Host "  3. 顯示完整狀態 JSON"
        Write-Host "  4. 顯示最新 30 行原始日誌"
        Write-Host "  5. 開啟日誌資料夾"
        Write-Host "  B. 返回主選單"
        $Choice = (Read-Host "請選擇診斷指令 [B]").Trim().ToUpperInvariant()
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
                Write-Host "畫面已儲存：$Output" -ForegroundColor Green
            }
        } elseif ($Choice -eq "3") {
            try {
                Invoke-RestMethod `
                    -Uri "http://127.0.0.1:$StatusPort/status" `
                    -TimeoutSec 2 |
                    ConvertTo-Json -Depth 8
            } catch {
                Write-Host "狀態 API 目前無法使用：$($_.Exception.Message)" -ForegroundColor Yellow
            }
        } elseif ($Choice -eq "4") {
            $LogFile = Join-Path $LogRoot ((Get-Date -Format "yyyyMMdd") + ".log")
            if (Test-Path $LogFile) {
                Get-Content -LiteralPath $LogFile -Tail 30
            } else {
                Write-Host "今天尚未產生日誌檔案。" -ForegroundColor Yellow
            }
        } elseif ($Choice -eq "5") {
            New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
            Start-Process explorer.exe -ArgumentList $LogRoot
        } else {
            Write-Host "無法辨識這個診斷指令。" -ForegroundColor Yellow
        }
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Dino Mutant Bot - 中文互動控制台 " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "執行位置：$RuntimeRoot"
Write-Host "AI 狀態接口：http://127.0.0.1:$StatusPort/status（僅限本機、唯讀）"

try {
    if (-not $SkipDoctor) {
        Invoke-EnvironmentCheck
    }
    $ExistingBots = Find-RunningBot
    if ($ExistingBots.Count -gt 0) {
        $ProcessIds = ($ExistingBots | ForEach-Object { $_.ProcessId }) -join ", "
        throw "已有另一個 Bot 正在執行（PID：$ProcessIds）。請先停止它再啟動新版。"
    }

    $Settings = Resolve-SpeedSettings -RequestedSpeed $Speed
    $SelectedSpeedName = if ($Settings.DisplaySpeed -eq "safe") { "安全" } elseif ($Settings.DisplaySpeed -eq "custom") { "自訂" } else { "快速" }
    Write-Host "已選擇速度：$SelectedSpeedName" -ForegroundColor Green
    if ($DryRun) {
        Write-Host "模擬檢查通過，未啟動 Bot 視窗。" -ForegroundColor Green
        exit 0
    }

    $BotProcess = Start-BotLogWindow $Settings
    try {
        while ($true) {
            Write-Host ""
            Write-Host "操作：[S]狀態 [T]調整速度 [D]診斷工具 [A]AI 接口 [R]重新啟動 [Q]停止離開" -ForegroundColor Cyan
            $Command = (Read-Host "請輸入指令 [S]").Trim().ToUpperInvariant()
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
                Write-Host "無法辨識這個指令。" -ForegroundColor Yellow
            }
        }
    } finally {
        Stop-BotLogWindow $BotProcess
    }
} catch {
    Write-Host ""
    Write-Host "錯誤：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
