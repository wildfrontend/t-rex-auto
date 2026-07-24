#!/bin/bash

set -euo pipefail

experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${DINO_BOT_ANDROID_USB_RUNTIME:-${experiment_root}/.runtime-android-usb}"
speed="${1:-safe}"
status_port="${2:-8865}"

"${experiment_root}/scripts/deploy-macos.sh" "${runtime_root}"
exec "${runtime_root}/start-bot.command" "${speed}" "${status_port}"
