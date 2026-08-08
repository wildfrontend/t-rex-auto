# 猛龍計畫

以 BlueStacks 5 為執行環境的可擴充 Python Bot Framework。核心採用
`Sense → Think → Act → Verify` 回饋循環，不依賴錄製 Macro。

目前完成 Auto Hunt MVP：辨識恐龍、選擇最大隊伍、發動狩獵並驗證結果。後續功能以
Feature 方式加入，不需要修改核心狀態機。

目前版本：`v0.0.6`。這一版會在親代數值無法辨識時自動保存完整畫面、六個數值 ROI
的原圖與二值化放大圖，並記錄原始字串與失敗次數，方便後續改善字模與裁切位置；同時
保留 v0.0.5 的容量連續讀值、`5/6/8` 字形校正、階段化巢穴掃描與未完成蛋驗證。
親代素質仍有 HP 升級 10–30、攻擊升級 1–3、速度有效值 1–150 的防呆，超出範圍時
拒絕替換；同時保留低效能電腦的孵化恢復、
慢速模式轉場容錯、多世代壓縮日誌、錨點／供給量規劃、
卡死逃生、半解析度比對，以及可調整狩獵速度及本機 AI 狀態接口：

- 使用者只需雙擊 `start-dashboard.cmd`；首次啟動會安裝 runtime，Bot 模式由網頁介面選擇。
- 孵蛋使用獨立的 `start-hatch-bot.cmd`，固定啟動 hatch feature，不會落入狩獵流程。
- 一個視窗顯示原始即時 LOG，另一個繁體中文互動視窗提供統計、調速、重啟與診斷工具。
- `127.0.0.1:8765` 提供結構化狀態與白名單控制接口，讓同一台電腦上的 AI 安全操作。
- Repository 內附 `.agents/skills/control-dino-bot`，限制 AI 使用固定接口與控制命令。
- 控制視窗按 `E` 會輸出經過敏感資訊遮蔽的診斷 ZIP，不需要提供遠端控制權。
- 診斷包包含環境檢查、最新工作階段、近期日誌、有效設定及 Codex 分析指引；截圖必須另外明確選擇。
- Windows 啟動器預設使用 `fast` 模式，也可由使用者切換 `safe` 或自訂毫秒數。
- CLI 可個別覆寫選恐龍、狩獵、確認及空轉掃描延遲。
- 黑畫面期間暫停 Detect/Verify，畫面恢復後才繼續原操作。
- 有定義下一個 UI 的重要按鈕，必須真的看到預期 UI 才算成功。
- 畫面只剩我方藍色路徑時，仍會累計空轉並安全重置地圖。
- 狩獵計數延後到確認按鈕驗證成功後才提交。

## 執行架構

```text
WSL /home/louis/github/wildfrontend/t-rex-auto
  ├─ 原始碼、Git、離線測試
  └─ scripts/windows/deploy-windows.sh
               │
               ▼
Windows D:\DinoMutantBot
  ├─ python\   可攜式 Python 3.12 runtime
  └─ app\      WSL 原始碼的執行副本
               │
               ├─ ADB framebuffer 背景擷取（預設）
               └─ Android SDK adb 執行 tap/swipe/long press
```

BlueStacks 必須保留在 Windows，不需要也不應安裝到 WSL。

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
- 狩獵確認因信箱已滿而反覆失敗時，自動關閉狩獵視窗、清空信箱並恢復狩獵。
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

### 1. BlueStacks

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

### 2. Windows runtime

專案使用不需管理員權限的可攜式 Python 3.12。若需要重建：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\windows\install-windows-runtime.ps1
```

### 3. 從 WSL 部署

每次修改程式碼或 detector assets 後執行。第二個參數可把既有 Python runtime 一併
封裝成可直接分享的完整資料夾：

```bash
bash scripts/windows/deploy-windows.sh
bash scripts/windows/deploy-windows.sh /mnt/d/DinoMutantBot-release /mnt/d/DinoMutantBot/python
```

### 4. 環境檢查

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

發佈包解壓後只需要雙擊：

```text
D:\DinoMutantBot\start-dashboard.cmd
```

Dashboard 會在目前 CMD 視窗前景執行並開啟瀏覽器；純狩獵、孵蛋＋狩獵、
安全停止、重啟與診斷都從網頁操作。關閉 CMD 視窗或按 `Ctrl+C` 即可停止
Dashboard，不建立登入啟動項或隱藏 watcher。

`fast` 是預設值，使用 300/900/1200 ms 的選恐龍、狩獵、確認期限；
`safe` 使用 1500/5000/3000 ms，適合反應較慢的電腦。

完整日誌保存在 `app\logs\YYYYMMDD.log`。
Bot 執行期間會阻止 Windows 系統睡眠，但不阻止螢幕依電源設定自動關閉；Bot
停止後會自動解除保持喚醒要求。

### 本機 AI 狀態與安全控制接口

Bot 執行時只監聽 `127.0.0.1`。查詢端點為唯讀，控制端只接受固定的安全動作：

```text
http://127.0.0.1:8765/health
http://127.0.0.1:8765/status
http://127.0.0.1:8765/actions
http://127.0.0.1:8765/settings
POST http://127.0.0.1:8765/control/stop
POST http://127.0.0.1:8765/control/restart-game
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
$control-dino-bot 請重新啟動 Dino Mutant App
```

Skill 只允許 `status/start/stop/restart/restart-game/doctor/diagnostics/snapshot`。啟動、停止、
重啟 Bot 或重啟遊戲 App 都必須由使用者當次明確要求，控制腳本也會強制檢查 `-Confirm`；
不允許 AI 自行執行 ADB 點擊、掃描 Port 或探索遊戲。`restart-game` 只會重啟設定中固定的
Dino Mutant package，不接受外部 package、activity 或 ADB 指令。控制腳本會先驗證
`/health` 服務身分；控制前還會確認 API PID、Port 占用者與 Bot 命令列一致，驗證失敗時
不會送出控制請求。

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
- `event_log.backup_count` / `log_backup_count`: 事件流與文字日誌各保留幾個舊世代，
  預設 20 與 12。事件流 16 MB 上限約 97 分鐘寫滿，只留一代等於只保得住約三小時，
  通宵執行隔天早上要看時前面幾小時已經被覆蓋。第二代以後會 gzip（實測壓到 6.5%
  與 4.4%），所以 20 代事件只佔約 52 MB。最新的一代刻意不壓縮，診斷包的
  `events-*.jsonl` 與 `20*.log` 兩個 glob 才能照舊運作，也不會誤把壓縮檔當文字讀。
  世代編號愈大愈舊（`.1` 最新）。
- `planner.max_center_distance_px`: 恐龍離畫面中心超過多少像素就不點，預設 600，設 0 停用。
  點下恐龍會讓地圖置中到牠身上，所以這個距離就是地圖要移動的量；移動愈大，那一下愈常
  只把地圖拉過去而沒有打開狩獵面板。實測 161 分鐘 1251 次點擊：300 px 內成功率 86%、
  300–500 px 69%、500 px 外只剩 21%，600 px 外的 31 次點擊只換到 2 次狩獵。調小會更
  保守（白工更少，但供給不足時會更早觸發重置）。被擋下的候選在 `plan` 事件裡記為
  `center_distance`。
- `planner.stalled_recenter_seconds`: 在採集地圖連續多少秒沒有安全目標後重置視野，預設 10。
  用秒數而非幀數，是因為一次掃描的成本會隨主機負載在 1080–3668 ms 之間浮動，同樣「4 幀」
  在忙碌的機器上是等 15 秒、在空閒的機器上只有 4.3 秒。
- `planner.recenter_min_candidates`: 通過所有拒絕規則的恐龍少於幾隻就重置地圖,預設 1。
  回中是補貨動作——它存在的理由是讓接下來幾次偵查都有足夠的恐龍可選,中央蛋只是
  「重置完成」的訊號。所以觸發條件看的是供給量,不是「這個 cycle 有沒有挑出目標」。
  調高會更早重置(地圖更滿,但重置次數變多),`stalled_recenter_seconds` 則控制供給
  不足要撐多久才真的重置——自己的狩獵路線會隨著隊伍返回而消失,不必一掉就重置。
- `planner.blind_idle_seconds`: 規劃器連續多少秒既選不出目標、也不在任何有期限的等待中，
  就強制解除所有卡住的階段，預設 20。其他每個逃生條件都寫成「看到某個控制項才放行」——
  `stalled_recenter_seconds` 要地圖地標、信箱流程的解除條件也要——而這種卡死的定義正好是
  那些控制項一個都看不到，所以它們全都不會觸發。實測一輪 42% 的時間卡在這種畫面上。
  這個計時器不依賴畫面提供任何東西，觸發時同時把當下的畫面存到 `logs/stalls/`。
- `planner.mail_stage_timeout_seconds`: 收信流程停止推進多少秒後放棄本輪，預設 20。
  期限從「上一次階段推進」起算而不是從進入信箱起算，所以只會砍掉不動的流程，不會砍掉慢的。
- `stalls.snapshots_enabled` / `snapshot_limit` / `snapshot_min_interval_seconds`:
  卡死畫面要不要存、留幾張、最短間隔幾秒，預設 `true` / 10 / 60。一段卡死每 20 秒會重報一次，
  沒有間隔下限的話一段三分鐘的卡死就會用九張幾乎一樣的圖洗掉全部保留額度。
- `planner.stage_scoped_scan`: 規劃階段只掃目前階段用得到的素材，程式預設 `true`；目前
  `config.json` 設為 `false`，使用較慢但每輪完整辨識的 Full Scan 穩定模式。登入／裝置
  紀錄／開場優惠三個對話框佔一次全掃描的四分之一，而它們跑起來之後不可能再出現；實測地圖
  階段因此省 42%、信箱流程省 56%。設為 `false` 可回到每個 cycle 都掃全部。
- `planner.full_scan_after_idle_cycles`: 連續幾個 cycle 規劃不出目標就把下一次掃描放回全部，
  預設 `2`。窄掃描漏看的東西長得跟空地圖一模一樣，這個計數就是察覺的方式；設 2 表示意外的
  對話框最多浪費一個 cycle。
- `planner.full_scan_interval_seconds`: 就算一路順利，最長多久也要全掃一次，預設 30 秒。
- `assets/manifest.json` 的 `match_scale`: 每個素材要在多少解析度下搜尋，預設 `1.0`。
  matchTemplate 的成本與搜尋範圍的像素數成正比、與素材大小和命中數無關，所以砍半是接近
  四倍的加速。實測 17 個素材在半解析度下信心值全部保住（`INTER_AREA` 縮圖等於低通濾波，
  把干擾比對的高頻雜訊去掉了），只有 18×18 的 `dinosaur` 標籤太小、維持 `1.0`。
- `capture.viewport`: Android 畫面在 BlueStacks client 內的 `[x,y,width,height]`；
  若含有 BlueStacks 側欄，應設定此值以確保 ADB 座標精準。
- `click_delay`: 一般點擊後條件式驗證的最長等待毫秒數；成功時會立即往下執行。
- `post_action_delays`: 各類操作等待下一個 UI 的最長期限，不是固定睡眠時間。
- `transition_poll_interval`: 等待期間重新擷取與辨識的間隔。
- `verify.minimum_checks`: 慢速電腦即使超過時間期限，最少仍會完成的驗證次數。
- `--speed safe|fast`: 從終端切換保守或快速延遲預設。
- `--status-port`: 本機狀態與白名單控制 API 連接埠；`0` 代表停用。
- `--dinosaur-delay-ms`、`--hunt-button-delay-ms`、`--hunt-confirm-delay-ms`、
  `--idle-delay-ms`: 以毫秒個別覆寫狩獵流程速度。
- `assets/manifest.json` 的 template `scales`: 同一辨識素材要嘗試的縮放倍率。
- `verify_retry`: 初次失敗後最多重試次數。
- `max_actions`: `0` 代表不限，用於 Debug 時建議先設為 `1`。
- `planner.recenter_every`: 完成多少次狩獵後重返主頁並重新置中，預設 `10`。
- `planner.own_path_radius`: 藍色虛線周圍的禁點半徑，預設 `90` px。
- `planner.mail_after_hunts`: 累積多少次狩獵後收取信箱，預設 `30`。
- `planner.capacity_wait_seconds`: 同時派出隊伍達 `10/10` 時的等待秒數，預設
  `300` 秒。
- `post_action_delays.target_too_strong`: 關閉過強目標後的等待時間，預設
  `300000` ms（5 分鐘）。
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
