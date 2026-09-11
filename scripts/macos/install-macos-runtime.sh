#!/bin/bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app_root="$(cd -- "${script_dir}/.." && pwd)"
venv_root="${app_root}/.venv"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This installer is for macOS."
  exit 1
fi

# Prefer 3.12/3.13: opencv-python-headless ships prebuilt wheels for them,
# while newer interpreters may trigger a slow source build that needs Xcode.
python_command=""
for candidate in "${DINO_BOT_PYTHON:-}" python3.12 python3.13 python3; do
  [[ -z "${candidate}" ]] && continue
  if command -v "${candidate}" >/dev/null 2>&1 &&
    "${candidate}" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' 2>/dev/null; then
    python_command="${candidate}"
    break
  fi
done

if [[ -z "${python_command}" ]]; then
  echo "ERROR: Python 3.12 or newer was not found."
  echo "Install Python (建議: brew install python@3.13), then run this script again."
  exit 1
fi

if ! "${python_command}" -c 'import sys; raise SystemExit(sys.version_info >= (3, 14))'; then
  echo "WARNING: $(${python_command} --version) 尚無 OpenCV 預編譯套件，安裝可能需要編譯很久。"
  echo "建議先安裝 Python 3.12 或 3.13（brew install python@3.13）再重新執行。"
fi

if [[ ! -x "${venv_root}/bin/python" ]]; then
  echo "Creating macOS Python environment at ${venv_root}"
  "${python_command}" -m venv "${venv_root}"
fi

"${venv_root}/bin/python" -m pip install --upgrade pip
"${venv_root}/bin/python" -m pip install --editable "${app_root}"

echo "macOS runtime ready: ${venv_root}/bin/python"
