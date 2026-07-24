#!/bin/bash

set -euo pipefail

experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${DINO_BOT_ANDROID_USB_RUNTIME:-${experiment_root}/.runtime-android-usb}"
status_port="${1:-8865}"
stop_command="${runtime_root}/stop-bot.command"

if [[ ! -x "${stop_command}" ]]; then
  echo "Android USB 實驗執行版尚未部署或沒有執行。"
  exit 1
fi

exec "${stop_command}" "${status_port}"
