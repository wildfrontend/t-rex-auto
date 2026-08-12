#!/usr/bin/env bash
# 打包 Windows 可攜版:Python runtime 與相依套件都隨包附上,解壓縮後雙擊即可
# 執行,不需要網路、不需要安裝。與 Lite 版的差別只有這一點。
# 預設輸出到 ~/Downloads,下載的 runtime 素材快取在 .runtime-windows/。
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# 版號的唯一來源;pyproject.toml 以 dynamic version 讀同一行。
version="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "${project_root}/src/dino_bot/__init__.py")"
if [[ -z "${version}" ]]; then
  echo "找不到版號:src/dino_bot/__init__.py 的 __version__" >&2
  exit 1
fi

python_version="3.12.10"
python_archive_name="python-${python_version}-embed-amd64.zip"
python_archive_url="https://www.python.org/ftp/python/${python_version}/${python_archive_name}"
# 與 install-windows-runtime.ps1 同一組素材,所以沿用同一個 MD5。
python_archive_md5="fe8ef205f2e9c3ba44d0cf9954e1abd3"

output_root="${1:-${HOME}/Downloads}"
package_name="DinoMutantBot-v${version}-Windows-Portable"
package_root="${output_root}/${package_name}"
package_app="${package_root}/app"
package_python="${package_root}/python"
cache_root="${project_root}/.runtime-windows"

host_python="${project_root}/.venv/bin/python"
if [[ ! -x "${host_python}" ]]; then
  host_python="$(command -v python3)"
fi

mkdir -p "${cache_root}"

# --- 1. 取得 Windows 內嵌版 Python ------------------------------------------
python_archive="${cache_root}/${python_archive_name}"
if [[ ! -f "${python_archive}" ]]; then
  echo "下載 ${python_archive_name} ..."
  curl -fsSL --retry 3 -o "${python_archive}.part" "${python_archive_url}"
  mv "${python_archive}.part" "${python_archive}"
fi
# macOS 內建 bash 是 3.2,沒有 ${var,,};打包腳本要能在原廠 shell 上跑。
actual_md5="$(
  { md5 -q "${python_archive}" 2>/dev/null || md5sum "${python_archive}" | cut -d' ' -f1; } \
    | tr '[:upper:]' '[:lower:]'
)"
if [[ "${actual_md5}" != "${python_archive_md5}" ]]; then
  echo "Python 壓縮檔校驗失敗:期望 ${python_archive_md5},實得 ${actual_md5}" >&2
  echo "快取檔案已刪除,請重新執行。" >&2
  rm -f "${python_archive}"
  exit 1
fi

# --- 2. 取得 Windows 相依套件(在 macOS 上跨平台下載 win_amd64 輪子)-------
site_cache="${cache_root}/site-packages-${python_version}"
if [[ ! -d "${site_cache}" ]]; then
  echo "下載 Windows 相依套件 ..."
  tmp_site="${site_cache}.part"
  rm -rf "${tmp_site}"
  "${host_python}" -m pip install --quiet --target "${tmp_site}" \
    --platform win_amd64 --python-version "${python_version%.*}" \
    --only-binary=:all: \
    "numpy>=2,<3" "opencv-python-headless>=4.10,<5" "mss>=9,<11" "pywin32>=306"

  # 這個專案只用 imread/imwrite 與矩陣運算,沒有任何 VideoCapture/CascadeClassifier。
  # 影片編解碼 DLL 與 haarcascade 資料合計約 38MB,對使用者是純粹的下載成本。
  rm -f "${tmp_site}"/cv2/opencv_videoio_ffmpeg*.dll
  rm -rf "${tmp_site}"/cv2/data
  # pywin32 只用到 win32gui/win32api/win32con/win32process(視窗擷取後端)。
  # COM、ISAPI、ADO 與 Pythonwin IDE 都不在路徑上,說明文件同理。
  rm -rf "${tmp_site}"/pythonwin "${tmp_site}"/adodbapi "${tmp_site}"/isapi \
         "${tmp_site}"/win32com "${tmp_site}"/win32comext "${tmp_site}"/PyWin32.chm
  find "${tmp_site}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  mv "${tmp_site}" "${site_cache}"
fi

# --- 3. 組裝發佈資料夾 -------------------------------------------------------
rm -rf "${package_root}" "${package_root}.zip"
mkdir -p "${package_app}/scripts" "${package_python}"

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
  "${project_root}/scripts/windows/cleanup-runtime-windows.ps1" \
  "${project_root}/scripts/windows/doctor-windows.ps1" \
  "${project_root}/scripts/windows/launcher-windows.ps1" \
  "${project_root}/scripts/windows/control-windows.ps1" \
  "${project_root}/scripts/windows/watch-running-bot.ps1" \
  "${package_app}/scripts/"

# 可攜版清空 adb.serial:收到這個包的人用哪一款模擬器、哪一個埠都不知道,
# 寫死一個埠等於保證一部分人打開就是「No ready ADB device」。留空會在啟動時
# 探測已知連接埠,恰好找到一台就用它;找到多台仍然要人選,不會替他猜。
"${host_python}" - "${package_app}/config.json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
config.setdefault("adb", {})["serial"] = None
with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY

cp -a "${project_root}/scripts/windows/start-dashboard-portable.cmd" \
      "${package_root}/start-dashboard.cmd"
cp -a "${project_root}/instances.json" "${package_root}/instances.json"
cp -a "${project_root}/使用教學.md" "${package_root}/使用教學.md"
cp -a "${project_root}/新手啟動指南.md" "${package_root}/新手啟動指南.md"
if [ -d "${project_root}/.agents" ]; then
  cp -a "${project_root}/.agents" "${package_root}/"
fi

# --- 4. 放入 runtime ---------------------------------------------------------
"${host_python}" -m zipfile -e "${python_archive}" "${package_python}"
cp -a "${project_root}/scripts/windows/python312._pth" "${package_python}/python312._pth"
mkdir -p "${package_python}/Lib/site-packages"
cp -a "${site_cache}/." "${package_python}/Lib/site-packages/"

find "${package_root}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "${package_root}" -name ".DS_Store" -delete 2>/dev/null || true

# --- 5. 出貨前檢查 -----------------------------------------------------------
# 在 macOS 上跑不動 Windows 的 python.exe,所以改為驗證每一個載入必需品都在位。
missing=0
for required in \
  "${package_python}/python.exe" \
  "${package_python}/python312.zip" \
  "${package_python}/python312._pth" \
  "${package_python}/Lib/site-packages/numpy/__init__.py" \
  "${package_python}/Lib/site-packages/cv2/cv2.pyd" \
  "${package_python}/Lib/site-packages/mss/__init__.py" \
  "${package_python}/Lib/site-packages/win32/lib/pywin32_bootstrap.py" \
  "${package_python}/Lib/site-packages/pywin32_system32/pywintypes312.dll" \
  "${package_root}/start-dashboard.cmd" \
  "${package_app}/main.py" \
  "${package_app}/config.json" ; do
  if [[ ! -e "${required}" ]]; then
    echo "出貨前檢查失敗,缺少:${required#${package_root}/}" >&2
    missing=1
  fi
done
# 批次檔以 LF 換行時,cmd.exe 會在 `^` 續行與跨行 if 區塊上解析錯誤,
# 使用者看到的就是雙擊後視窗一閃就關。這一項不能只靠 .gitattributes 保證。
if ! LC_ALL=C tr -dc '\r' < "${package_root}/start-dashboard.cmd" | grep -q .; then
  echo "出貨前檢查失敗:start-dashboard.cmd 不是 CRLF 換行" >&2
  missing=1
fi
if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

if command -v zip >/dev/null 2>&1; then
  (cd "${output_root}" && zip -qr "${package_name}.zip" "${package_name}")
else
  (cd "${output_root}" && "${host_python}" -m zipfile -c "${package_name}.zip" "${package_name}")
fi
echo "已打包:${package_root}"
echo "已打包:${package_root}.zip"
