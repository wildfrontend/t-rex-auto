@echo off
setlocal
title 猛龍計畫 - Beginner Auto Hatch

set "hatch_runner=%~dp0app\scripts\run-hatch-windows.ps1"
if not exist "%hatch_runner%" (
  echo ERROR: Hatch runner not found: %hatch_runner%
  pause
  exit /b 1
)

echo BEGINNER AUTO HATCH
echo Hatch every ready egg, then choose All, auto-place once, and collect once.
echo The incubator-full notice is handled safely without repeating collection.
echo This mode never sorts parents, enters the cave, or removes dinosaurs.
echo Start this launcher from the normal home screen.
echo.

set "hatch_mode=%~1"
if "%hatch_mode%"=="" set "hatch_mode=debug"

set "hatch_speed=%~2"
if "%hatch_speed%"=="" set "hatch_speed=safe"

set "hatch_status_port=%~3"
if "%hatch_status_port%"=="" set "hatch_status_port=8775"

set "hatch_max_actions=%~4"
if "%hatch_max_actions%"=="" set "hatch_max_actions=0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%hatch_runner%" ^
  -Feature "hatch-beginner" ^
  -Mode "%hatch_mode%" ^
  -Speed "%hatch_speed%" ^
  -StatusPort "%hatch_status_port%" ^
  -MaxActions "%hatch_max_actions%" ^
  -MaxCycles "0"
set "hatch_exit_code=%ERRORLEVEL%"

echo.
echo Beginner Auto Hatch closed with exit code %hatch_exit_code%.
pause
exit /b %hatch_exit_code%
