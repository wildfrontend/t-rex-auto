@echo off
chcp 65001 >nul 2>&1
setlocal

rem 上面的 chcp 排在所有非 ASCII 內容之前:cmd.exe 以「當下的字碼頁」解析批次檔,
rem 切換前輸出的中文會變成亂碼。本檔為 UTF-8 無 BOM(有 BOM 會讓第一行解析失敗)。
title Dino Mutant Bot - Dashboard

rem 可攜版啟動器:Python runtime 已隨包附上,不下載、不安裝。
rem 每一條失敗路徑都要留下訊息並寫入 log。使用者回報的「雙擊後一閃就關」
rem 沒有留下任何線索,無訊息的離開比失敗本身更難處理。
rem 關鍵訊息中英並陳:萬一字碼頁切換在某些環境失效,英文那行仍可讀。

set "runtime_root=%~dp0"
if "%runtime_root:~-1%"=="\" set "runtime_root=%runtime_root:~0,-1%"
set "log_dir=%runtime_root%\logs"
set "log_file=%log_dir%\launcher.log"
if not exist "%log_dir%" mkdir "%log_dir%" >nul 2>&1

echo ---- launcher start %DATE% %TIME% ---->>"%log_file%"
echo runtime_root=%runtime_root%>>"%log_file%"

rem 從 ZIP 內直接雙擊時,Windows 會把檔案解到唯讀暫存資料夾,相對路徑隨即失效。
rem 直接用字串置換檢查暫存路徑，避免建立額外的命令管線。
if not "%runtime_root:AppData\Local\Temp=%"=="%runtime_root%" (
  set "reason=Running from inside the ZIP. Extract the whole folder first, then double-click."
  set "reason_zh=偵測到是直接從 ZIP 裡執行。請先把整個資料夾解壓縮到例如桌面,再雙擊 start-dashboard.cmd。"
  goto :fail
)

set "python_exe=%runtime_root%\python\python.exe"
if not exist "%python_exe%" (
  set "reason=Bundled Python is missing. The ZIP was not fully extracted."
  set "reason_zh=找不到內建 Python,這個包沒有完整解壓縮。請重新解壓縮整個 ZIP。"
  goto :fail
)

set "main_script=%runtime_root%\app\main.py"
if not exist "%main_script%" (
  set "reason=Entry point app\main.py is missing. The ZIP was not fully extracted."
  set "reason_zh=找不到程式進入點 app\main.py,這個包沒有完整解壓縮。"
  goto :fail
)

rem 內建 runtime 在打包時已驗證檔案齊全,這裡確認它在這台機器上真的載得起來。
"%python_exe%" -c "import encodings, numpy, cv2, mss" >>"%log_file%" 2>&1
if errorlevel 1 (
  set "reason=Bundled Python cannot load its modules; see logs\launcher.log. Antivirus quarantine is the usual cause."
  set "reason_zh=內建 Python 無法載入必要模組,詳細錯誤在 logs\launcher.log。最常見的原因是防毒軟體隔離了 DLL。"
  goto :fail
)

set "dashboard_port=%~1"
if "%dashboard_port%"=="" set "dashboard_port=8780"
echo port=%dashboard_port%>>"%log_file%"

echo.
echo 猛龍計畫儀表板正在啟動：http://127.0.0.1:%dashboard_port%
echo (關閉本視窗或按 Ctrl+C 即停止儀表板)
echo.

cd /d "%runtime_root%"
"%python_exe%" "%main_script%" --config "%runtime_root%\app\config.json" dashboard --port %dashboard_port% --open-browser
set "exit_code=%ERRORLEVEL%"
echo dashboard exited with %exit_code%>>"%log_file%"

if not "%exit_code%"=="0" (
  echo.
  echo Dashboard exited with code %exit_code%.
  echo 儀表板結束,離開碼 %exit_code%。記錄檔:%log_file%
  pause
)
endlocal
exit /b 0

:fail
echo.
echo ============================================
echo  Cannot start / 無法啟動
echo ============================================
echo %reason%
echo %reason_zh%
echo.
echo Log: %log_file%
echo FAILED: %reason%>>"%log_file%"
echo.
pause
endlocal
exit /b 1
