# Dino Mutant Bot

以 BlueStacks 5 為執行環境的可擴充 Python Bot Framework。核心採用
`Sense → Think → Act → Verify` 回饋循環，不依賴錄製 Macro。

目前完成 Auto Hunt MVP：辨識恐龍、選擇最大隊伍、發動狩獵並驗證結果。後續功能以
Feature 方式加入，不需要修改核心狀態機。

## 執行架構

```text
WSL /home/louis/github/wildfrontend/t-rex-auto
  ├─ 原始碼、Git、離線測試
  └─ scripts/deploy-windows.sh
               │
               ▼
Windows %LOCALAPPDATA%\DinoMutantBot
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
- OpenCV Template Matching、HSV 輪廓偵測與 NMS。
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

## 專案結構

```text
.
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
  -File scripts\install-windows-runtime.ps1
```

### 3. 從 WSL 部署

每次修改程式碼或 detector assets 後執行：

```bash
bash scripts/deploy-windows.sh
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
C:\Users\Louis\AppData\Local\DinoMutantBot\python\python.exe `
  C:\Users\Louis\AppData\Local\DinoMutantBot\app\main.py `
  --config C:\Users\Louis\AppData\Local\DinoMutantBot\app\config.json `
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

工具會裁切圖片並自動更新 `assets/manifest.json`。同一物件若有多種動畫或縮放，
應加入多張 template。也可以在 manifest 使用 HSV 偵測：

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

先用 Debug 模式限制一次操作：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run-windows.ps1 -Mode debug -MaxActions 1
```

確認 `debug/` 中的前後圖片與結果正確後再執行 Runtime：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run-windows.ps1 -Mode runtime
```

執行一批 10 次狩獵，完成返回主頁與森林置中後停止：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run-windows.ps1 -Mode runtime -BatchSize 10 -MaxCycles 1
```

Training 模式：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run-windows.ps1 -Mode training
```

按 `Ctrl+C` 安全停止。

## 擷取效能測試

```powershell
C:\Users\Louis\AppData\Local\DinoMutantBot\python\python.exe `
  C:\Users\Louis\AppData\Local\DinoMutantBot\app\main.py `
  --config C:\Users\Louis\AppData\Local\DinoMutantBot\app\config.json `
  benchmark --frames 100
```

結果必須至少達到 `config.json` 的 `capture_fps`（預設 10 FPS）。

## 設定重點

- `capture.backend`: `adb` 不搶 focus；`mss` 較快但會把 BlueStacks 拉到前景。
- `capture.viewport`: Android 畫面在 BlueStacks client 內的 `[x,y,width,height]`；
  若含有 BlueStacks 側欄，應設定此值以確保 ADB 座標精準。
- `click_delay`: 點擊到驗證畫面的等待毫秒數。
- `post_action_delays`: 可針對確認按鈕等動畫較長的操作設定額外等待時間。
- `verify_retry`: 初次失敗後最多重試次數。
- `max_actions`: `0` 代表不限，用於 Debug 時建議先設為 `1`。
- `planner.recenter_every`: 完成多少次狩獵後重返主頁並重新置中，預設 `10`。
- `planner.own_path_radius`: 藍色虛線周圍的禁點半徑，預設 `90` px。
- `workflow.max_cycles`: 完整批次數；`0` 代表持續執行。

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
