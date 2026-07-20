#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_app="/mnt/c/Users/Louis/AppData/Local/DinoMutantBot/app"

mkdir -p "${runtime_app}"
cp -a \
  "${project_root}/main.py" \
  "${project_root}/config.json" \
  "${project_root}/pyproject.toml" \
  "${project_root}/src" \
  "${project_root}/assets" \
  "${runtime_app}/"

echo "Deployed to ${runtime_app}"
