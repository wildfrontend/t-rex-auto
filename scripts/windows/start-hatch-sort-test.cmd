@echo off
setlocal
title Dino Mutant Bot - Hatch Sort Test

set "hatch_runner=%~dp0app\scripts\run-hatch-windows.ps1"
if not exist "%hatch_runner%" (
  echo ERROR: Hatch runner not found: %hatch_runner%
  pause
  exit /b 1
)

echo.
echo SAFE TEST: Select Dino all-tags and attack-descending sort.
echo Open Select Dino by tapping one nest parent before continuing.
echo This test will not select or replace any dinosaur.
echo.
pause

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%hatch_runner%" ^
  -Feature "hatch-sort-test" ^
  -Mode "debug" ^
  -Speed "safe" ^
  -StatusPort "8768" ^
  -MaxActions "0" ^
  -MaxCycles "0"
set "hatch_exit_code=%ERRORLEVEL%"

echo.
echo Hatch sort test closed with exit code %hatch_exit_code%.
pause
exit /b %hatch_exit_code%
