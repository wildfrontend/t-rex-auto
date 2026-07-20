#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_app="/mnt/c/Users/Louis/AppData/Local/DinoMutantBot/app"
launcher_dir="/mnt/d/DinoMutantBot"

mkdir -p "${runtime_app}" "${runtime_app}/scripts" "${launcher_dir}"
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
  "${runtime_app}/scripts/"
cp -a "${project_root}/scripts/start-bot.cmd" "${launcher_dir}/start-bot.cmd"
rm -f -- "${runtime_app}/scripts/start-bot.cmd"

echo "Deployed to ${runtime_app}"
