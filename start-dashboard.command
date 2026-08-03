#!/bin/zsh
# Dino Mutant Bot - macOS 儀表板啟動器(雙擊執行)
cd "$(dirname "$0")" || exit 1

if [ ! -x .venv/bin/python ]; then
  echo "找不到 .venv/bin/python,請先建立虛擬環境並安裝相依套件。"
  read -r "?按 Enter 關閉..."
  exit 1
fi

echo "Dino 儀表板啟動中:http://127.0.0.1:8780"
echo "(關閉本視窗或按 Ctrl+C 即停止儀表板)"
exec .venv/bin/python -m dino_bot.cli --config config-mac.json dashboard --port 8780 --open-browser
