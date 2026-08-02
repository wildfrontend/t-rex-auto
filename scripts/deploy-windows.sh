#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${1:-/mnt/d/DinoMutantBot-App}"
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
# Emulator processes can keep adb.exe and its DLLs open on Windows. The bundled
# toolchain is immutable for an app update, so preserve installed files and only
# fill in anything that is missing.
cp -a --update=none "${project_root}/tools" "${runtime_app}/"
cp -a \
  "${project_root}/scripts/run-windows.ps1" \
  "${project_root}/scripts/run-hatch-windows.ps1" \
  "${project_root}/scripts/run-dashboard-windows.ps1" \
  "${project_root}/scripts/doctor-windows.ps1" \
  "${project_root}/scripts/launcher-windows.ps1" \
  "${project_root}/scripts/control-windows.ps1" \
  "${project_root}/scripts/install-windows-runtime.ps1" \
  "${project_root}/scripts/setup-windows.ps1" \
  "${project_root}/scripts/python312._pth" \
  "${project_root}/scripts/watch-running-bot.ps1" \
  "${runtime_app}/scripts/"
cp -a "${project_root}/scripts/start-bot.cmd" "${runtime_root}/start-bot.cmd"
cp -a "${project_root}/scripts/start-dashboard.cmd" "${runtime_root}/start-dashboard.cmd"
cp -a "${project_root}/scripts/start-hatch-hunt.cmd" "${runtime_root}/start-hatch-hunt.cmd"

# Keep the portable root limited to the three user-facing entrypoints. Older
# development launchers remain recoverable under app/scripts instead of being
# deleted from an existing installation.
legacy_launchers="${runtime_app}/scripts/legacy-launchers"
mkdir -p "${legacy_launchers}"
for launcher in \
  start-hatch-bot.cmd \
  start-hatch-full.cmd \
  start-hatch-filter-test.cmd \
  start-hatch-sort-test.cmd \
  start-hatch-parent-test.cmd \
  start-hatch-attack-test.cmd \
  start-hatch-hp-test.cmd; do
  if [[ -f "${runtime_root}/${launcher}" ]]; then
    mv -f "${runtime_root}/${launcher}" "${legacy_launchers}/${launcher}"
  fi
done
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
