#!/usr/bin/env bash
# 打包 Windows Lite 發佈 ZIP(不含 Python runtime;首次啟動由
# install-windows-runtime.ps1 下載)。預設輸出到 ~/Downloads。
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# 版號的唯一來源;pyproject.toml 以 dynamic version 讀同一行。
version="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "${project_root}/src/dino_bot/__init__.py")"
if [[ -z "${version}" ]]; then
  echo "找不到版號:src/dino_bot/__init__.py 的 __version__" >&2
  exit 1
fi
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
  "${project_root}/config-s13.json" \
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

# --- 出貨前檢查:Windows 腳本必須是 CRLF -------------------------------------
# 批次檔以 LF 換行時,cmd.exe 會在 `^` 續行與跨行 if 區塊上解析錯誤,使用者看到
# 的就是雙擊後視窗一閃就關。.gitattributes 已宣告 eol=crlf,但屬性是後來才加的,
# 早於它 checkout 的檔案會在工作目錄留著舊的 LF,打包腳本原樣複製就把問題出貨。
# 所以這裡檢查的是實際要打包的 bytes,而不是信任 git 設定。
missing=0
while IFS= read -r script; do
  if ! LC_ALL=C tr -dc '\r' < "${script}" | grep -q .; then
    echo "出貨前檢查失敗,不是 CRLF 換行:${script#${package_root}/}" >&2
    echo "  修法:rm '${script#${package_root}/}' 後在專案內 git checkout 同名檔案" >&2
    missing=1
  fi
done < <(find "${package_root}" \( -name "*.cmd" -o -name "*.ps1" \) -type f)
if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

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
