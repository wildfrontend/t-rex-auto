#!/bin/bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app_root="$(cd -- "${script_dir}/.." && pwd)"
venv_root="${app_root}/.venv"
python_command="${DINO_BOT_PYTHON:-python3}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This installer is for macOS."
  exit 1
fi

if ! command -v "${python_command}" >/dev/null 2>&1; then
  echo "ERROR: Python 3.12 or newer was not found."
  echo "Install Python, then run this script again."
  exit 1
fi

if ! "${python_command}" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'; then
  echo "ERROR: $(${python_command} --version) is too old; Python 3.12 or newer is required."
  exit 1
fi

if [[ ! -x "${venv_root}/bin/python" ]]; then
  echo "Creating macOS Python environment at ${venv_root}"
  "${python_command}" -m venv "${venv_root}"
fi

"${venv_root}/bin/python" -m pip install --upgrade pip
"${venv_root}/bin/python" -m pip install --editable "${app_root}"

echo "macOS runtime ready: ${venv_root}/bin/python"
