@echo off
setlocal
title Dino Mutant Bot - close this window to stop

echo Dino Mutant Bot
echo Close this terminal window to stop the bot.
echo Logs are also saved under app\logs.
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-windows.ps1" -Mode runtime -MaxCycles 0 -BatchSize 10 -MailAfterHunts 30
set "bot_exit_code=%ERRORLEVEL%"

echo.
echo Bot stopped with exit code %bot_exit_code%.
pause
exit /b %bot_exit_code%
