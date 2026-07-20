@echo off
setlocal
title Dino Mutant Bot - close this window to stop

echo Dino Mutant Bot
echo Close this terminal window to stop the bot.
echo Logs are also saved under app\logs.
echo.

set "bot_runner=%LOCALAPPDATA%\DinoMutantBot\app\scripts\run-windows.ps1"
if not exist "%bot_runner%" (
  echo ERROR: Bot runner not found: %bot_runner%
  pause
  exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%bot_runner%" -Mode runtime -MaxCycles 0 -BatchSize 10 -MailAfterHunts 30
set "bot_exit_code=%ERRORLEVEL%"

echo.
echo Bot stopped with exit code %bot_exit_code%.
pause
exit /b %bot_exit_code%
