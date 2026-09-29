@echo off
setlocal
title 猛龍計畫 - Auto Hatch + Hunt

set "combined_runner=%~dp0app\scripts\run-hatch-windows.ps1"
if not exist "%combined_runner%" (
  echo ERROR: Combined runner not found: %combined_runner%
  pause
  exit /b 1
)

echo AUTO HATCH + HUNT
echo Runs the complete Hatch workflow, then hunts during the egg cooldown.
echo The final 30 seconds are reserved for a safe return to the centered home map.
echo Start this launcher from the normal home screen.
echo.

set "combined_mode=%~1"
if "%combined_mode%"=="" set "combined_mode=runtime"

set "combined_speed=%~2"
if "%combined_speed%"=="" set "combined_speed=fast"

set "combined_status_port=%~3"
if "%combined_status_port%"=="" set "combined_status_port=8773"

set "combined_max_actions=%~4"
if "%combined_max_actions%"=="" set "combined_max_actions=0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%combined_runner%" ^
  -Feature "hatch-hunt" ^
  -Mode "%combined_mode%" ^
  -Speed "%combined_speed%" ^
  -StatusPort "%combined_status_port%" ^
  -MaxActions "%combined_max_actions%" ^
  -MaxCycles "0"
set "combined_exit_code=%ERRORLEVEL%"

echo.
echo Auto Hatch + Hunt closed with exit code %combined_exit_code%.
pause
exit /b %combined_exit_code%
