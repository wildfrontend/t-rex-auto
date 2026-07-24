#!/bin/bash

set -euo pipefail

runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app_root="${runtime_root}/app"
runtime_python="${app_root}/.venv/bin/python"
speed="${1:-fast}"
status_port="${2:-8765}"

cd "${app_root}"

if [[ ! -x "${runtime_python}" ]]; then
  echo "第一次啟動：正在建立 macOS 執行環境。"
  "${app_root}/scripts/install-macos-runtime.sh"
fi

echo
echo "正在檢查 Python、ADB、裝置與辨識素材……"
if ! "${runtime_python}" "${app_root}/main.py" \
  --config "${app_root}/config.json" doctor; then
  echo
  echo "環境檢查失敗，Bot 未啟動。"
  if [[ -t 0 ]]; then
    read -r -p "按 Enter 關閉視窗……" _
  fi
  exit 1
fi

echo
"${runtime_python}" "${app_root}/scripts/control-macos.py" \
  start \
  --speed "${speed}" \
  --status-port "${status_port}" \
  --confirm

status_output="$(mktemp -t dino-bot-status)"
trap 'rm -f "${status_output}"' EXIT
bot_ready=0
for _ in {1..15}; do
  if "${runtime_python}" "${app_root}/scripts/control-macos.py" \
    status \
    --status-port "${status_port}" >"${status_output}" 2>&1; then
    bot_ready=1
    break
  fi
  sleep 1
done

echo
cat "${status_output}"
if ((bot_ready == 0)); then
  echo
  echo "Bot 未能在 15 秒內提供狀態，請檢查：${app_root}/logs/macos-launcher.log"
  if [[ -f "${app_root}/logs/macos-launcher.log" ]]; then
    tail -n 20 "${app_root}/logs/macos-launcher.log"
  fi
  if [[ -t 0 ]]; then
    read -r -p "按 Enter 關閉視窗……" _
  fi
  exit 1
fi

echo
echo "Bot 已在背景執行。日誌：${app_root}/logs/macos-launcher.log"
if [[ -t 0 ]]; then
  read -r -p "按 Enter 關閉此視窗（Bot 會繼續執行）……" _
fi
