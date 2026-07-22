---
name: control-dino-bot
description: Safely inspect and operate the local Dino Mutant Bot through its allowlisted status API and control-windows.ps1 entrypoint. Use when the user asks for hunting progress, current Bot status, failures, recent actions, health checks, screenshots, environment diagnostics, starting, stopping, restarting, changing the fast/safe launch profile, or using a non-default local status port. Never use this skill for arbitrary ADB actions, game exploration, or unrequested process control.
---

# Control Dino Mutant Bot

Use only the Bot's structured localhost API and the fixed Windows controller. Keep all access on
`127.0.0.1`; never expose the service to a LAN or public address.

## Resolve the controller

Resolve the skill directory, then go up three directories to get `BOT_ROOT`. Use the first existing
path below without searching elsewhere:

1. `BOT_ROOT/app/scripts/control-windows.ps1` for a deployed Bot folder.
2. `BOT_ROOT/scripts/control-windows.ps1` for a source checkout.

If neither path exists, stop and report that the Bot controller is missing. In WSL, convert this
exact path with `wslpath -w` before passing it to `powershell.exe`; do not scan the filesystem.

Use this command shape:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File <control-windows.ps1> -Action <action> -StatusPort <port>
```

Use port `8765` unless the user gives another port or the interactive launcher reports a different
one. Never probe or scan ports. If the API is unavailable, report the attempted URL and ask the user
for the configured port.

## Read-only requests

For progress, status, failure, stuck, black-screen, settings, or recent-action questions, run:

```powershell
... -Action status -StatusPort 8765
```

Treat `successful_hunts` as confirmed hunts. Report `current_stage`, `successful_hunts`,
`total_actions`, `verification_failures`, `black_screen_detections`, `game_restarts`, and
`last_successful_hunt`. Do not infer that the Bot is stuck from one snapshot alone; use timestamps
and request another status check if the last log may still be advancing.

Run `-Action doctor` only when the user asks to diagnose prerequisites or connectivity. Run
`-Action snapshot` only when the user explicitly asks for a current screenshot; report the returned
file path.

## State-changing requests

Only start, stop, or restart when the user explicitly requests that action in the current turn.
Never infer permission from a status request, a failure, a black screen, or an earlier conversation.

The controller enforces confirmation. Pass `-Confirm` only after verifying explicit intent:

```powershell
... -Action start   -Speed fast -StatusPort 8765 -Confirm
... -Action stop                -StatusPort 8765 -Confirm
... -Action restart -Speed fast -StatusPort 8765 -Confirm
```

Allow only `fast` or `safe`. Use the user's stated profile; otherwise preserve the known current
profile, or use `fast` only for a new start when no current profile is known. A restart may take up
to 20 seconds. After a start or restart, query status once and report the result.

For custom millisecond timings or changing the port interactively, direct the user to the Chinese
control window: `[T]` changes timings and `[P]` changes the local API port. Do not edit
`config.json` or source code as a substitute for a runtime control request. Port cleanup is also a
human-only launcher action: tell the user to use `[K]` when it is offered and enter the displayed
confirmation token themselves; never reproduce that action with process commands.

## Allowed HTTP surface

Use only these loopback routes:

- `GET /health`
- `GET /status`
- `GET /actions`
- `GET /settings`
- `POST /control/stop`, only after explicit stop or restart intent

Do not try other routes, methods, parameters, hosts, or payloads.

## Hard boundaries

- Do not run `adb`, `taskkill`, `Stop-Process`, or arbitrary shell commands.
- Do not click, tap, swipe, or explore the game UI directly.
- Do not change source files, configuration, templates, or detector assets.
- Do not expose, tunnel, or bind the API beyond `127.0.0.1`.
- Do not guess ports, runtime folders, credentials, or device identifiers.
- Stop on `confirmation_required`, `status_api_unavailable`, or an unknown response; report it
  instead of finding another route.

## Report results

Answer in the user's language. State the action performed, port used, whether it succeeded, and the
key status counts. Mention that the interface is local-only when explaining connection behavior.
