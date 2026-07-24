#!/bin/bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${1:-${project_root}/.runtime-macos}"
runtime_root="$(mkdir -p "${runtime_root}" && cd -- "${runtime_root}" && pwd)"
runtime_app="${runtime_root}/app"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This deployer is for macOS." >&2
  exit 1
fi

if [[ "${runtime_root}" == "/" || "${runtime_root}" == "${HOME}" ]]; then
  echo "ERROR: Refusing unsafe runtime path: ${runtime_root}" >&2
  exit 1
fi

if [[ "${runtime_root}" == "${project_root}" ]]; then
  echo "ERROR: Runtime and source directories must be different." >&2
  exit 1
fi

mkdir -p \
  "${runtime_app}/scripts" \
  "${runtime_app}/logs" \
  "${runtime_app}/debug" \
  "${runtime_app}/capture" \
  "${runtime_app}/diagnostics"

for source_file in \
  main.py \
  capture.py \
  detector.py \
  planner.py \
  action.py \
  verify.py \
  config.py \
  config.json \
  pyproject.toml
do
  /bin/cp "${project_root}/${source_file}" "${runtime_app}/${source_file}"
done

/usr/bin/rsync -a --delete "${project_root}/src/" "${runtime_app}/src/"
/usr/bin/rsync -a --delete "${project_root}/assets/" "${runtime_app}/assets/"

for script_name in \
  control-macos.py \
  install-macos-runtime.sh \
  run-macos.sh
do
  /bin/cp \
    "${project_root}/scripts/${script_name}" \
    "${runtime_app}/scripts/${script_name}"
done

/bin/cp \
  "${project_root}/scripts/start-runtime-macos.sh" \
  "${runtime_root}/start-bot.command"
/bin/cp \
  "${project_root}/stop-bot.command" \
  "${runtime_root}/stop-bot.command"

/bin/chmod +x \
  "${runtime_root}/start-bot.command" \
  "${runtime_root}/stop-bot.command" \
  "${runtime_app}/scripts/install-macos-runtime.sh" \
  "${runtime_app}/scripts/run-macos.sh"

echo "macOS runtime deployed: ${runtime_app}"
