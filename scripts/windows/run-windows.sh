#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-runtime}"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "${script_dir}/run-windows.ps1" -Mode "${mode}"
