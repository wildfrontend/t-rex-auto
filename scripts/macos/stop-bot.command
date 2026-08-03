#!/bin/bash

set -u

launcher_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
status_port="${1:-8765}"

if [[ -f "${launcher_root}/app/scripts/control-macos.py" ]]; then
  controller="${launcher_root}/app/scripts/control-macos.py"
elif [[ -f "${launcher_root}/scripts/control-macos.py" ]]; then
  controller="${launcher_root}/scripts/control-macos.py"
else
  echo "找不到 macOS Bot 控制器，未執行任何關閉操作。"
  exit_code=1
  if [[ -t 0 ]]; then
    read -r -p "按 Enter 關閉視窗……" _
  fi
  exit "${exit_code}"
fi

echo "正在安全關閉 Dino Mutant Bot（Port ${status_port}）……"
python3 "${controller}" \
  stop \
  --status-port "${status_port}" \
  --confirm
exit_code=$?

if ((exit_code == 0)); then
  echo "Bot 已確認關閉。"
else
  echo "Bot 未能安全關閉；請查看上方錯誤訊息。"
fi

if [[ -t 0 ]]; then
  read -r -p "按 Enter 關閉視窗……" _
fi
exit "${exit_code}"
