#!/usr/bin/env bash
# 打包 Windows Lite 發佈 ZIP(不含 Python runtime;首次啟動由
# install-windows-runtime.ps1 下載)。預設輸出到 ~/Downloads。
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "${project_root}/pyproject.toml")"
output_root="${1:-${HOME}/Downloads}"
package_name="DinoMutantBot-v${version}-Windows-Lite"
package_root="${output_root}/${package_name}"
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
  "${project_root}/config.json" \
  "${project_root}/pyproject.toml" \
  "${project_root}/src" \
  "${project_root}/assets" \
  "${project_root}/tools" \
  "${package_app}/"

cp -a \
  "${project_root}/scripts/windows/run-windows.ps1" \
  "${project_root}/scripts/windows/run-hatch-windows.ps1" \
  "${project_root}/scripts/windows/run-dashboard-windows.ps1" \
  "${project_root}/scripts/windows/uninstall-windows.ps1" \
  "${project_root}/scripts/windows/cleanup-runtime-windows.ps1" \
  "${project_root}/scripts/windows/doctor-windows.ps1" \
  "${project_root}/scripts/windows/launcher-windows.ps1" \
  "${project_root}/scripts/windows/control-windows.ps1" \
  "${project_root}/scripts/windows/install-windows-runtime.ps1" \
  "${project_root}/scripts/windows/setup-windows.ps1" \
  "${project_root}/scripts/windows/python312._pth" \
  "${project_root}/scripts/windows/watch-running-bot.ps1" \
  "${package_app}/scripts/"

cp -a "${project_root}/scripts/windows/start-dashboard.cmd" "${package_root}/start-dashboard.cmd"
cp -a "${project_root}/instances.json" "${package_root}/instances.json"
cp -a "${project_root}/使用教學.md" "${package_root}/使用教學.md"
cp -a "${project_root}/新手啟動指南.md" "${package_root}/新手啟動指南.md"
if [ -d "${project_root}/.agents" ]; then
  cp -a "${project_root}/.agents" "${package_root}/"
fi

find "${package_root}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "${package_root}" -name ".DS_Store" -delete 2>/dev/null || true

if command -v zip >/dev/null 2>&1; then
  (cd "${output_root}" && zip -qr "${package_name}.zip" "${package_name}")
elif command -v python3 >/dev/null 2>&1; then
  (cd "${output_root}" && python3 -m zipfile -c "${package_name}.zip" "${package_name}")
else
  echo "Neither zip nor python3 is available to create the release archive." >&2
  exit 1
fi
echo "已打包:${package_root}"
echo "已打包:${package_root}.zip"
