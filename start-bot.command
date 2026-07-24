#!/bin/bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${DINO_BOT_MACOS_RUNTIME:-${project_root}/.runtime-macos}"

"${project_root}/scripts/deploy-macos.sh" "${runtime_root}"
exec "${runtime_root}/start-bot.command" "$@"
