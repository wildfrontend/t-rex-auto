@echo off
setlocal
title 猛龍計畫 - Full Auto Hatch

set "hatch_runner=%~dp0app\scripts\run-hatch-windows.ps1"
if not exist "%hatch_runner%" (
  echo ERROR: Hatch runner not found: %hatch_runner%
  pause
  exit /b 1
)

echo FULL AUTO HATCH
echo Hatch eggs, optimize Attack and HP parents, auto-place Top and Mass,
echo collect eggs, then cull in the cave only when dinosaur count is above 320.
echo If no incubator egg is ready, collect all nest eggs before the cooldown.
echo Incubator scanning stops immediately when the visible top rows have no ready egg.
echo Start this launcher from the normal home screen.
echo.

set "hatch_mode=%~1"
if "%hatch_mode%"=="" set "hatch_mode=debug"

set "hatch_speed=%~2"
if "%hatch_speed%"=="" set "hatch_speed=safe"

set "hatch_status_port=%~3"
if "%hatch_status_port%"=="" set "hatch_status_port=8772"

set "hatch_max_actions=%~4"
if "%hatch_max_actions%"=="" set "hatch_max_actions=0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%hatch_runner%" ^
  -Feature "hatch-full" ^
  -Mode "%hatch_mode%" ^
  -Speed "%hatch_speed%" ^
  -StatusPort "%hatch_status_port%" ^
  -MaxActions "%hatch_max_actions%" ^
  -MaxCycles "0"
set "hatch_exit_code=%ERRORLEVEL%"

echo.
echo Full Auto Hatch closed with exit code %hatch_exit_code%.
pause
exit /b %hatch_exit_code%
