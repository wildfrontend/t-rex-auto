@echo off
setlocal
title 猛龍計畫 - Hatch

set "hatch_runner=%~dp0app\scripts\run-hatch-windows.ps1"
if not exist "%hatch_runner%" (
  echo ERROR: Hatch runner not found: %hatch_runner%
  pause
  exit /b 1
)

set "hatch_mode=%~1"
if "%hatch_mode%"=="" set "hatch_mode=runtime"

set "hatch_speed=%~2"
if "%hatch_speed%"=="" set "hatch_speed=safe"

set "hatch_status_port=%~3"
if "%hatch_status_port%"=="" set "hatch_status_port=8766"

set "hatch_max_actions=%~4"
if "%hatch_max_actions%"=="" (
  if /I "%hatch_mode%"=="debug" (
    set "hatch_max_actions=1"
  ) else (
    set "hatch_max_actions=0"
  )
)

set "hatch_max_cycles=%~5"
if "%hatch_max_cycles%"=="" set "hatch_max_cycles=0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%hatch_runner%" ^
  -Mode "%hatch_mode%" ^
  -Speed "%hatch_speed%" ^
  -StatusPort "%hatch_status_port%" ^
  -MaxActions "%hatch_max_actions%" ^
  -MaxCycles "%hatch_max_cycles%"
set "hatch_exit_code=%ERRORLEVEL%"

echo.
echo Hatch runner closed with exit code %hatch_exit_code%.
pause
exit /b %hatch_exit_code%
