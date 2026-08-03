#!/usr/bin/env bash
# 打包 macOS 發佈資料夾與 ZIP(預設輸出到 ~/Downloads)。
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "${project_root}/pyproject.toml")"
output_root="${1:-${HOME}/Downloads}"
package_root="${output_root}/DinoMutantBot-v${version}-macOS"
package_app="${package_root}/app"

rm -rf "${package_root}" "${package_root}.zip"
mkdir -p "${package_app}/scripts"

cp -a \
  "${project_root}/main.py" \
  "${project_root}/capture.py" \
  "${project_root}/detector.py" \
  "${project_root}/planner.py" \
  "${project_root}/action.py" \
  "${project_root}/verify.py" \
  "${project_root}/config.py" \
  "${project_root}/pyproject.toml" \
  "${project_root}/src" \
  "${project_root}/assets" \
  "${package_app}/"
# macOS 包的預設設定即 Mac 版設定(BlueStacks Air 的 ADB 5555)。
cp -a "${project_root}/config-mac.json" "${package_app}/config.json"

cp -a \
  "${project_root}/scripts/macos/install-macos-runtime.sh" \
  "${project_root}/scripts/macos/control-macos.py" \
  "${project_root}/scripts/macos/run-macos.sh" \
  "${package_app}/scripts/"

cp -a "${project_root}/scripts/macos/start-bot.command" "${package_root}/start-bot.command"
cp -a "${project_root}/scripts/macos/stop-bot.command" "${package_root}/stop-bot.command"
cp -a "${project_root}/使用教學.md" "${package_root}/使用教學.md"

# 儀表板啟動器(套件版:venv 建在 app/.venv,第一次啟動會自動安裝)。
cat >"${package_root}/start-dashboard.command" <<'LAUNCHER'
#!/bin/bash
set -euo pipefail
runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app_root="${runtime_root}/app"
runtime_python="${app_root}/.venv/bin/python"

if [[ ! -x "${runtime_python}" ]]; then
  echo "第一次啟動:正在建立 macOS 執行環境。"
  "${app_root}/scripts/macos/install-macos-runtime.sh"
fi

cd "${app_root}"
echo "Dino 儀表板啟動中:http://127.0.0.1:8780"
echo "(關閉本視窗或按 Ctrl+C 即停止儀表板)"
exec "${runtime_python}" "${app_root}/main.py" \
  --config "${app_root}/config.json" dashboard --port 8780 --open-browser
LAUNCHER
chmod +x "${package_root}/start-dashboard.command"

find "${package_root}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "${package_root}" -name ".DS_Store" -delete 2>/dev/null || true

(cd "${output_root}" && zip -qr "DinoMutantBot-v${version}-macOS.zip" "DinoMutantBot-v${version}-macOS")
echo "已打包:${package_root}"
echo "已打包:${package_root}.zip"
