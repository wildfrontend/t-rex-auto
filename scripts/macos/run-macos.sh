#!/bin/bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app_root="$(cd -- "${script_dir}/.." && pwd)"
python_executable="${app_root}/.venv/bin/python"
speed="${1:-fast}"
status_port="${2:-8765}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This launcher is for macOS."
  exit 1
fi
if [[ ! -x "${python_executable}" ]]; then
  echo "ERROR: macOS runtime is not installed."
  echo "Run ${script_dir}/install-macos-runtime.sh first."
  exit 1
fi
if [[ "${speed}" != "fast" && "${speed}" != "safe" ]]; then
  echo "ERROR: Speed must be fast or safe."
  exit 2
fi
if [[ ! "${status_port}" =~ ^[0-9]+$ ]] ||
  ((status_port < 1 || status_port > 65535)); then
  echo "ERROR: Status port must be between 1 and 65535."
  exit 2
fi

echo "Bot speed profile: ${speed}"
echo "Local status API: http://127.0.0.1:${status_port}/status"
echo "System sleep is blocked while the Bot runs; display sleep remains enabled."

cd "${app_root}"
exec /usr/bin/caffeinate -i \
  "${python_executable}" \
  "${app_root}/main.py" \
  --config "${app_root}/config.json" \
  run \
  --mode runtime \
  --speed "${speed}" \
  --status-port "${status_port}" \
  --verbose
