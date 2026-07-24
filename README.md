# Dino Mutant Bot

以 BlueStacks 5 為執行環境的可擴充 Python Bot Framework。核心採用
`Sense → Think → Act → Verify` 回饋循環，不依賴錄製 Macro。

目前完成 Auto Hunt MVP：辨識恐龍、選擇最大隊伍、發動狩獵並驗證結果。後續功能以
Feature 方式加入，不需要修改核心狀態機。

目前版本：`v0.2.3`。這一版排除中央蛋附近的恐龍誤判，並改善地圖與信箱
轉場的驗證穩定度：

- macOS 雙擊 `start-bot.command` 會先部署到 `.runtime-macos/app`，再從獨立副本啟動。
- 動作後每 100–250 ms 檢查下一個 UI；成功時立即繼續，不必等滿固定秒數。
- 驗證階段只掃描預期下一個 UI 與必要錯誤提示，不再重跑全部辨識素材。
- 已確認的「恐龍 → 狩獵 → 確認」可沿用同一驗證畫面，省去重複截圖及完整掃描。
- 確認狩獵回到地圖後直接沿用該畫面選下一隻，不等待隊伍回程或再次完整掃描。
- 點恐龍未進入狩獵畫面時立即釋放等待狀態，直接尋找下一個安全目標。
- 中央蛋周圍 50px 不會被選為恐龍，避免重複點擊固定畫面中心。
- 地圖與信箱按鈕使用 2.5–3 秒最大轉場時間，畫面提早就緒仍立即繼續。
- 轉場上限從第一個辨識結果開始計算，避免辨識本身耗時造成虛假失敗。
- Capture、Detect 與 Verify 日誌會記錄實際耗時，方便繼續定位效能瓶頸。
- `fast`、`safe` 的所有延遲與輪詢速度都集中在 `config.json` 的 `speed_profiles`。
- macOS 使用 `caffeinate` 在 Bot 執行期間防止系統睡眠，仍允許螢幕休眠。
- Windows 與 macOS 都使用 ADB framebuffer，不需要搶走滑鼠或鍵盤焦點。
- 使用者只需雙擊 `start-bot.cmd`；啟動器會先檢查 Python、ADB、素材及畫面擷取。
- 一個視窗顯示原始即時 LOG，另一個繁體中文互動視窗提供統計、調速、重啟與診斷工具。
- `127.0.0.1:8765` 提供結構化狀態與白名單停止接口，讓同一台電腦上的 AI 安全操作。
- Repository 內附 `.agents/skills/control-dino-bot`，限制 AI 使用固定接口與控制命令。
- 控制視窗按 `E` 會輸出經過敏感資訊遮蔽的診斷 ZIP，不需要提供遠端控制權。
- 診斷包包含環境檢查、最新工作階段、近期日誌、有效設定及 Codex 分析指引；截圖必須另外明確選擇。
- Windows 啟動器預設使用 `fast` 模式，也可互動切換 `safe` 或自訂毫秒數。
- CLI 可個別覆寫選恐龍、狩獵、確認及空轉掃描延遲。
- 黑畫面期間暫停 Detect/Verify，畫面恢復後才繼續原操作。
- 有定義下一個 UI 的重要按鈕，必須真的看到預期 UI 才算成功。
- 畫面只剩我方藍色路徑時，仍會累計空轉並安全重置地圖。
- 狩獵計數延後到確認按鈕驗證成功後才提交。

## 執行架構

```text
Windows
  ├─ BlueStacks 5
  ├─ 可攜式 Python 3.12 runtime
  └─ PowerShell 啟動／控制器

macOS（Apple Silicon）
  ├─ BlueStacks Air
  ├─ .runtime-macos/app 執行副本與獨立 .venv
  └─ Finder .command 啟動／Python 安全控制器

兩個平台
  ├─ ADB framebuffer 背景擷取（預設）
  └─ Android SDK adb 執行 tap/swipe/long press
```

Windows 使用 BlueStacks 5；macOS 使用 BlueStacks Air。WSL 只用於 Windows 版的
原始碼與離線測試，不執行 BlueStacks。

## 已完成項目

- BlueStacks 視窗自動尋找：支援視窗標題及 `HD-Player.exe` 程序辨識。
- MSS 指定客戶區域擷取：畫面以 BGR `numpy.ndarray` 留在 RAM。
- ADB framebuffer 擷取備援。
- OpenCV Template Matching（支援單一素材多尺寸比對）、HSV 輪廓偵測與 NMS。
- 最近畫面中心、最高信心值兩種 Planner 策略。
- ADB tap、swipe、long press、sleep 與畫面/裝置座標映射。
- 操作後重新 Capture、Detect、Verify；失敗會重新感知與規劃，最多重試三次。
- 獨立 State：Idle、Capture、Detect、Planning、Action、Verify、Recover、Stopped。
- Runtime、Debug、Training 三種模式。
- Runtime 不寫圖片或影片；操作日誌寫入 `logs/YYYYMMDD.log`。
- Debug 每次操作保存 `Before.png`、`After.png`、`debug.json`。
- Training 以 1–5 FPS 保存，超過設定上限時刪除最舊圖片。
- OpenCV template 製作 CLI。
- Windows 環境診斷及擷取 FPS benchmark。
- 狩獵時自動選擇最大群組，並依序完成狩獵與確認按鈕。
- 已點過的目標不會在同一畫面重複選取；重複狩獵警告會中止該次操作。
- 偵測我方隊伍的藍色虛線，排除路徑及兩側 90 px 安全區內的恐龍。
- 每 10 次狩獵自動返回主頁，再從森林入口回到以中央蛋置中的採集地圖。
- 連續多幀沒有安全恐龍時，自動重置視野，不會放寬藍線保護或無限等待。
- 約 30 次狩獵後自動開啟信箱，依序執行「全部獲取、資源獲取、關閉」。
- 右上角同時派出隊伍為 `10/10` 時不再選目標，等待 5 分鐘後重試。
- 出現「目標太強了，你會輸」時關閉狩獵視窗並等待 5 分鐘。
- Unity 畫面持續全黑 45 秒時只重啟遊戲 App；短暫轉場不處理，且有 90 秒重啟冷卻。
- 黑畫面不會被當成「按鈕消失」或像素變化成功，避免重複點擊與虛假狩獵計數。
- 遊戲重啟後可優先處理重複登入、不同設備歷史記錄與啟動優惠提示。
- 自動關閉「自動成長結果」及其後續「自動戰鬥」快捷視窗，再回到採集地圖。

## 專案結構

```text
.
├── .agents/skills/control-dino-bot/
│   ├── SKILL.md
│   └── agents/openai.yaml
├── main.py
├── config.json
├── src/dino_bot/
│   ├── actions.py
│   ├── application.py
│   ├── assets.py
│   ├── capture.py
│   ├── cli.py
│   ├── config.py
│   ├── detection.py
│   ├── doctor.py
│   ├── engine.py
│   ├── interfaces.py
│   ├── logging.py
│   ├── models.py
│   ├── modes.py
│   ├── planning.py
│   ├── recovery.py
│   ├── status.py
│   ├── status_server.py
│   └── verification.py
├── assets/
│   ├── manifest.json
│   └── templates/
├── scripts/
├── tests/
├── logs/
├── debug/
└── capture/
```

根目錄的 `capture.py`、`detector.py`、`planner.py`、`action.py`、`verify.py`、
`config.py` 是需求文件介面的相容匯出；正式實作位於 `src/dino_bot`。

## 第一次設定

### macOS：BlueStacks Air

需求：

- Apple Silicon Mac 與 BlueStacks Air。
- Python 3.12 或更新版本。
- Android SDK Platform Tools；預設會尋找
  `~/Library/Android/sdk/platform-tools/adb`。
- 遊戲直向畫面 `900 × 1600`。

在 BlueStacks Air 設定中啟用 Android Debug Bridge，確認顯示
`127.0.0.1:5555`，啟動 Dino Mutant 並停在主地圖或採集地圖。接著在 Finder
雙擊根目錄的：

```text
start-bot.command
```

第一次執行會部署 `.runtime-macos/app`、建立其中的 `.venv` 並安裝相依套件。
若 macOS 阻止開啟，可在 Finder
對檔案按右鍵後選擇「打開」一次。

要明確關閉 Bot，可直接雙擊 `stop-bot.command`。它會驗證本機 API、Port、PID
及執行路徑，送出安全停止要求，並等待程序真正退出後才顯示完成。終端用法：

```bash
./stop-bot.command
./stop-bot.command 8877
```

終端控制命令：

```bash
python3 scripts/control-macos.py status
python3 scripts/control-macos.py doctor
python3 scripts/control-macos.py diagnostics
python3 scripts/control-macos.py snapshot
python3 scripts/control-macos.py stop --confirm
python3 scripts/control-macos.py restart --speed fast --confirm
```

預設狀態 Port 是 `8765`；非預設 Port 必須在每個控制命令加上
`--status-port <Port>`。來源端控制腳本會自動連到已部署的執行副本。背景啟動記錄
保存在 `.runtime-macos/app/logs/macos-launcher.log`，完整 Bot 日誌保存在
`.runtime-macos/app/logs/YYYYMMDD.log`。

原始碼與執行內容是分開的：修改 `src/` 或 `config.json` 不會直接改到正在運行的
程式；下次雙擊 `start-bot.command` 時才會重新部署。`.runtime-macos` 已加入
`.gitignore`，其中的日誌、診斷包、截圖與 Python 環境不會被部署流程刪除。

macOS 只支援 `capture.backend: "adb"`；`mss` 視窗擷取仍是 Windows 專用。

### Windows：BlueStacks 5

1. 啟動 BlueStacks 5。
2. 開啟「設定 → 進階」。
3. 啟用「Android 調試橋（ADB）」。
4. 確認畫面顯示 `127.0.0.1:5555`。
5. 保持 BlueStacks 視窗開啟且不要最小化。

專案預設使用：

```text
C:\Users\Louis\AppData\Local\Android\Sdk\platform-tools\adb.exe
127.0.0.1:5555
```

### Windows runtime

專案使用不需管理員權限的可攜式 Python 3.12。若需要重建：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\install-windows-runtime.ps1
```

### 從 WSL 部署

每次修改程式碼或 detector assets 後執行。第二個參數可把既有 Python runtime 一併
封裝成可直接分享的完整資料夾：

```bash
bash scripts/deploy-windows.sh
bash scripts/deploy-windows.sh /mnt/d/DinoMutantBot-release /mnt/d/DinoMutantBot/python
```

### Windows 環境檢查

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File scripts/doctor-windows.ps1
```

所有必要檢查應顯示 `PASS`。Detector 尚未放入素材時會顯示 `WARN`。

## 建立 Auto Collect 辨識素材

目前 Detector 使用每隻恐龍上方共同的 `Lv` 標籤作為錨點，再將點擊位置偏移到
恐龍身體，因此不限定恐龍種類。若遊戲字型或解析度改變，可重新建立錨點素材。

先將遊戲停在採集畫面，保存一張明確要求的開發截圖：

```powershell
D:\DinoMutantBot\python\python.exe `
  D:\DinoMutantBot\app\main.py `
  --config D:\DinoMutantBot\app\config.json `
  snapshot --output debug\collect-screen.png
```

根據截圖決定目標區域 `[x, y, width, height]`，再於 WSL 建立 template：

```bash
PYTHONPATH=/tmp/t-rex-auto-deps:src python3 main.py template \
  --input debug/collect-screen.png \
  --roi 100 200 48 48 \
  --type dinosaur \
  --name dinosaur-level-label \
  --threshold 0.60 \
  --click-offset 22 32
```

工具會裁切圖片並自動更新 `assets/manifest.json`。同一物件若只有 UI 縮放差異，
可以在單一 template 設定 `scales`；若圖案或動畫本身不同，則應加入多張 template：

```json
{
  "type": "device_history_confirm_button",
  "file": "templates/device-history-confirm-button.png",
  "threshold": 0.82,
  "scales": [0.88, 0.9, 0.95, 1.0],
  "click_offset": [86, 41]
}
```

`click_offset` 會依命中的 template 尺寸同步縮放。`scales` 必須是大於零的數值；未設定
時維持原本的 `1.0`。也可以在 manifest 使用 HSV 偵測：

```json
{
  "templates": [],
  "hsv_ranges": [
    {
      "type": "resource",
      "lower": [50, 180, 180],
      "upper": [75, 255, 255],
      "min_area": 100,
      "max_area": 5000
    }
  ]
}
```

## 執行

部署後只需要雙擊：

```text
D:\DinoMutantBot\start-bot.cmd
```

啟動流程會先做環境檢查，再開啟兩個視窗：

- `Dino Mutant Bot - Control`：互動控制、統計、調速、診斷與 AI API 資訊。
- Bot LOG 視窗：保留完整 Capture、Detect、Planning、Action、Verify、Recover 日誌。

互動視窗可使用：

```text
S  查詢成功狩獵、操作、失敗、黑屏、重啟及最近動作
T  改用 fast、safe 或自訂時間，並以新參數重啟 Bot
P  切換本機接口 Port；Port 被占用時會顯示程式與 PID
D  環境檢查、ADB 截圖、完整 JSON、原始日誌、開啟日誌資料夾
E  產生不含截圖的 Codex 診斷包並開啟輸出資料夾
A  顯示本機 AI API 端點
R  使用目前參數重啟
Q  停止 Bot 並關閉控制流程
```

也可在終端預先指定模式：

```bat
D:\DinoMutantBot\start-bot.cmd fast
D:\DinoMutantBot\start-bot.cmd safe
D:\DinoMutantBot\start-bot.cmd fast 8877
```

`fast` 使用 300/900/1200 ms 的選恐龍、狩獵、確認延遲；`safe` 則使用
1500/5000/3000 ms。

完整日誌保存在 `app\logs\YYYYMMDD.log`。
Bot 執行期間會阻止 Windows 系統睡眠，但不阻止螢幕依電源設定自動關閉；Bot
停止後會自動解除保持喚醒要求。

### 本機 AI 狀態與安全控制接口

Bot 執行時只監聽 `127.0.0.1`。查詢端點為唯讀，控制端只接受固定的安全停止動作：

```text
http://127.0.0.1:8765/health
http://127.0.0.1:8765/status
http://127.0.0.1:8765/actions
http://127.0.0.1:8765/settings
POST http://127.0.0.1:8765/control/stop
```

AI 或本機工具可直接讀取 `/status`，取得本次工作階段的成功狩獵數、信箱循環、
操作數、驗證失敗、黑屏、遊戲重啟、目前階段及最近操作。控制視窗按 `P` 可切換
Port。啟動或切換時若 Port 被占用，控制視窗會顯示占用程式、執行檔與 PID，預設選項
是改用下一個 Port。只有程序命令列與 `/health` API 身分都確認為 Dino Bot，且 API
回報 PID 與占用者一致時，才會顯示 `[K]` 清理選項；使用者輸入確認碼後還會再檢查
一次占用者，避免等待輸入期間 Port 已被其他程序接手。清理時必須手動
輸入畫面上的 `CLEAN-<Port>` 確認碼；若正常停止失敗，強制結束前還會要求第二次
`FORCE-<PID>` 確認。無法辨識或不是 Dino Bot 的程序不會被關閉。

使用 Codex 開啟 repository 或部署資料夾後，可直接說：

```text
$control-dino-bot 幫我查狩獵進度
$control-dino-bot 用 8877 Port 查詢目前狀態
$control-dino-bot 請停止 Bot
```

Skill 只允許 `status/start/stop/restart/doctor/diagnostics/snapshot`。啟動、停止及重啟必須由使用者
當次明確要求，控制腳本也會強制檢查 `-Confirm`；不允許 AI 自行執行 ADB 點擊、
掃描 Port 或探索遊戲。控制腳本會先驗證 `/health` 服務身分；停止與重啟還會確認
API PID、Port 占用者與 Bot 命令列一致，驗證失敗時不會送出控制請求。

不啟動 HTTP 服務也能從 CLI 查詢同一份結構化資料：

```powershell
D:\DinoMutantBot\python\python.exe D:\DinoMutantBot\app\main.py `
  --config D:\DinoMutantBot\app\config.json status --json
```

### Codex 診斷包

控制視窗按 `E` 可直接產生不含截圖的安全診斷包，輸出位置為：

```text
D:\DinoMutantBot\app\diagnostics\dino-diagnostic-YYYYMMDD-HHMMSS.zip
```

診斷選單 `D → 7` 才會加入目前遊戲畫面，選擇前會顯示明確提示。ZIP 內含
`summary.json`、`status.json`、`doctor.json`、遮蔽後的 `settings.json`、近期日誌及
`README_FOR_CODEX.md`。可直接把 ZIP 上傳給 Codex，請它說明故障原因、使用者可採取的
步驟，以及哪些問題需要修改 Bot 程式。診斷包不會建立遠端連線，也不包含任意控制接口。

也可以從終端產生：

```powershell
D:\DinoMutantBot\python\python.exe D:\DinoMutantBot\app\main.py `
  --config D:\DinoMutantBot\app\config.json diagnostics
```

只有在使用者同意分享畫面時才加上 `--include-screenshot`。

先用 Debug 模式限制一次操作：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run-windows.ps1 -Mode debug -MaxActions 1
```

確認 `debug/` 中的前後圖片與結果正確後再執行 Runtime：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run-windows.ps1 -Mode runtime -Speed fast
```

如果要個別調整，數值單位為毫秒，下次啟動即生效：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run-windows.ps1 -Mode runtime -Speed fast `
  -DinosaurDelayMs 800 -HuntButtonDelayMs 2500 `
  -HuntConfirmDelayMs 1800 -IdleDelayMs 150
```

執行完整流程：每 10 次重置地圖，累積 30 次後收取信箱並停止：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run-windows.ps1 -Mode runtime `
  -BatchSize 10 -MailAfterHunts 30 -MaxCycles 1
```

Training 模式：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run-windows.ps1 -Mode training
```

按 `Ctrl+C` 安全停止。

## 擷取效能測試

```powershell
D:\DinoMutantBot\python\python.exe `
  D:\DinoMutantBot\app\main.py `
  --config D:\DinoMutantBot\app\config.json `
  benchmark --frames 100
```

結果必須至少達到 `config.json` 的 `capture_fps`（預設 10 FPS）。

## 設定重點

- `capture.backend`: `adb` 不搶 focus；`mss` 較快但會把 BlueStacks 拉到前景。
- `planner.stalled_recenter_frames`: 連續多少幀沒有安全目標後重置視野，預設 8。
- `capture.viewport`: Android 畫面在 BlueStacks client 內的 `[x,y,width,height]`；
  若含有 BlueStacks 側欄，應設定此值以確保 ADB 座標精準。
- `click_delay`: 一般動作等待下一個 UI 的最長毫秒數，不是固定休眠。
- `transition_poll_interval`: 等待轉場期間重新擷取畫面的間隔，預設 `250` ms。
- `post_action_delays`: 各按鈕等待下一個 UI 的最長時間；畫面提早就緒便立即繼續。
- `speed_profiles`: `safe`、`fast` 的點擊、流程、空轉與輪詢設定唯一來源。
- `--speed safe|fast`: 從終端切換保守或快速延遲預設。
- `--status-port`: 本機狀態與白名單控制 API 連接埠；`0` 代表停用。
- `--dinosaur-delay-ms`、`--hunt-button-delay-ms`、`--hunt-confirm-delay-ms`、
  `--idle-delay-ms`、`--poll-interval-ms`: 以毫秒個別覆寫狩獵流程速度。
- `assets/manifest.json` 的 template `scales`: 同一辨識素材要嘗試的縮放倍率。
- `verify_retry`: 初次失敗後最多重試次數。
- `max_actions`: `0` 代表不限，用於 Debug 時建議先設為 `1`。
- `planner.recenter_every`: 完成多少次狩獵後重返主頁並重新置中，預設 `10`。
- `planner.own_path_radius`: 藍色虛線周圍的禁點半徑，預設 `90` px。
- `planner.anchor_exclusion_radius`: 中央蛋周圍不選恐龍的半徑，預設 `50` px。
- `planner.mail_after_hunts`: 累積多少次狩獵後收取信箱，預設 `30`。
- `planner.capacity_wait_seconds`: 同時派出隊伍達 `10/10` 時的等待秒數，預設
  `300` 秒。
- `post_action_delays.no_available_dinosaurs`: 關閉「沒有可用恐龍」提示後等待
  畫面改變的最長時間，預設 `300` ms。
- `post_action_delays.target_too_strong`: 關閉過強目標視窗的轉場上限，預設
  `3000` ms。
- `post_action_delays.map_exit_nest_button`、`forest_recenter_button` 與信箱流程：
  地圖／信箱動畫的最大轉場時間，預設 `2500–3000` ms。
- `planner.action_cooldowns_ms.target_too_strong`: 過強目標關閉並驗證成功後的
  可中斷冷卻，預設 `300000` ms（5 分鐘）。
- `recovery.black_screen_timeout_seconds`: 持續黑畫面多久後重啟遊戲，預設 `45` 秒。
- `recovery.restart_cooldown_seconds`: 兩次遊戲重啟的最短間隔，預設 `90` 秒。
- `workflow.max_cycles`: 完整「狩獵、信箱收取、關閉」流程次數；`0` 代表持續執行。

## 背景執行

預設使用 ADB framebuffer，因此 Bot 不使用滑鼠，也不需要 BlueStacks 是前景視窗；
可以讓其他視窗蓋住 BlueStacks並正常使用電腦。Windows 可以鎖定或關閉螢幕，
但不能進入睡眠或休眠，否則 Python、BlueStacks 與 ADB 都會暫停。

## 測試

WSL 缺少 `python3.12-venv` 時，可以使用獨立 dependency directory：

```bash
PYTHONPATH=/tmp/t-rex-auto-deps:src python3 -m pytest -p no:cacheprovider
PYTHONPATH=/tmp/t-rex-auto-deps:src python3 -m ruff check --no-cache .
```

若已安裝 `python3.12-venv`，則可使用標準 `.venv` 與 `pip install -e '.[dev]'`。

## 擴充功能

新增 Auto Mail、Hatching、Breeding、Battle 或 Titan 時，建立新的 Feature detector、
planner、workflow 與 verifier，並透過現有介面注入 `BotContext`。核心 Engine 與 State
不需要知道遊戲功能細節。
