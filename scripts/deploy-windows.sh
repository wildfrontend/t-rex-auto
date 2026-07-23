#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${1:-/mnt/d/DinoMutantBot}"
runtime_python_source="${2:-}"
runtime_app="${runtime_root}/app"

mkdir -p "${runtime_app}" "${runtime_app}/scripts"
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
  "${runtime_app}/"
cp -a \
  "${project_root}/scripts/run-windows.ps1" \
  "${project_root}/scripts/doctor-windows.ps1" \
  "${project_root}/scripts/launcher-windows.ps1" \
  "${project_root}/scripts/control-windows.ps1" \
  "${project_root}/scripts/install-windows-runtime.ps1" \
  "${project_root}/scripts/setup-windows.ps1" \
  "${project_root}/scripts/python312._pth" \
  "${project_root}/scripts/watch-running-bot.ps1" \
  "${runtime_app}/scripts/"
cp -a "${project_root}/scripts/start-bot.cmd" "${runtime_root}/start-bot.cmd"
cp -a "${project_root}/使用教學.md" "${runtime_root}/使用教學.md"
cp -a "${project_root}/.agents" "${runtime_root}/"

if [[ -n "${runtime_python_source}" ]]; then
  if [[ ! -f "${runtime_python_source}/python.exe" ]]; then
    echo "Python runtime source is invalid: ${runtime_python_source}" >&2
    exit 1
  fi
  mkdir -p "${runtime_root}/python"
  cp -a "${runtime_python_source}/." "${runtime_root}/python/"
fi

echo "Deployed portable app to ${runtime_root}"
