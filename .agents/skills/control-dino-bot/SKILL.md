---
name: control-dino-bot
description: Precisely inspect and safely operate the local Dino Mutant Bot through its mode-specific localhost status APIs, verified Windows process identity, allowlisted control-windows.ps1 entrypoint, and read-only shared-log fallback. Use for Bot running/stopped checks, old-process checks, hunting or hatch progress, current stage, recent actions, failures, health, diagnostics, screenshots, start/stop/restart, game restart, speed profile, WSL/Windows connectivity, or status-port questions. Never use for arbitrary ADB, game exploration, raw process killing, or unrequested state changes.
---

# Control Dino Mutant Bot

Use verified evidence. Never equate an unreachable API with a stopped process.

## Resolve mode and port first

Use an explicitly supplied port when present. Otherwise resolve the mode from the user's command,
the launcher they used, dashboard state, or the newest `Feature | ...` log line. Use this fixed map:

| Mode | Port |
|---|---:|
| `hunt` | 8765 |
| `hatch` | 8766 |
| `hatch-full` | 8772 |
| `hatch-hunt` | 8773 |
| `hatch-stage-*` | 8774 |

Port 8780 is the dashboard, not a Bot status API. Do not probe arbitrary ports. When asked whether
an old process remains, check only the old mode's known port and the intended mode's known port.

## Resolve the live controller

For the standard deployment use:

- Windows: `D:\DinoMutantBot-App\app\scripts\control-windows.ps1`
- WSL: `/mnt/d/DinoMutantBot-App/app/scripts/control-windows.ps1`

Use another runtime root only when the user explicitly supplies it. Use the source checkout's
`scripts/control-windows.ps1` only for source-development requests, not to control the deployed Bot.

In WSL, convert the exact controller path with `wslpath -w` and call:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File <control-windows.ps1> -Action <action> -StatusPort <resolved-port>
```

## Evidence ladder

1. Run the Windows controller. A successful `/health` identity check plus process/port ownership is
   authoritative for running/stopped and safe mutations.
2. If the controller returns an identity error, stop. Never control that port.
3. If WSL returns `Exec format error`, Windows interop is unavailable. Do not use WSL
   `127.0.0.1` as evidence about Windows loopback.
4. For a read-only status question, run the bundled fallback:

```bash
python3 <skill-root>/scripts/inspect_runtime.py \
  --runtime-root /mnt/d/DinoMutantBot-App --mode <resolved-mode>
```

Interpret fallback states exactly:

- `active_recently`: log activity within 30 seconds; strong activity evidence, no process identity.
- `stopped_by_log`: the last session marker is stop; historical evidence, no process identity.
- `unknown`: stale or incomplete evidence. Do not call it running or stopped.

For concurrent/old-process questions, log fallback cannot distinguish two writers. Report
`process_identity_verified=false` and require a Windows-side controller result for certainty.

## Read-only requests

For status, progress, failure, or recent-action requests run `-Action status` with the resolved port.
Report `current_stage`, `successful_hunts`, `total_actions`, `verification_failures`,
`black_screen_detections`, `game_restarts`, and `last_successful_hunt`. Treat only
`successful_hunts` as confirmed hunts. Use timestamps or a second snapshot before calling a run
stuck.

Run `doctor` only for prerequisite/connectivity diagnosis, `snapshot` only when the user requests a
current screenshot, and `diagnostics` only when the user requests a diagnostic bundle.

## State-changing requests

Only start, stop, restart, or restart the game when explicitly requested in the current turn. Pass
`-Confirm` only after that explicit request:

```powershell
... -Action start        -Speed fast -StatusPort <port> -Confirm
... -Action stop                     -StatusPort <port> -Confirm
... -Action restart      -Speed fast -StatusPort <port> -Confirm
... -Action restart-game             -StatusPort <port> -Confirm
```

Allow only `fast` or `safe`. Preserve a known profile; otherwise use `fast` for a new start. After
start/restart, query status once. A mutation requires the Windows controller and verified process
identity; never substitute log inference or file writes.

For custom timings or port changes direct the user to launcher `[T]` or `[P]`. Port cleanup `[K]`
and its displayed confirmation token are human-only.

## Allowed HTTP surface

Use only loopback routes `GET /health`, `/status`, `/actions`, `/settings`; and, after explicit user
authorization, `POST /control/stop` or `/control/restart-game`. Never bind or tunnel beyond
loopback.

## Hard boundaries

- Never run `adb`, `taskkill`, `Stop-Process`, raw taps/swipes, or arbitrary process commands.
- Never guess ports, runtime roots, package names, device IDs, or process identity.
- Never mutate source/config/templates as a substitute for runtime control.
- Stop mutations on `confirmation_required`, `status_api_unavailable`, any identity mismatch, or an
  unknown response.

## Report

State the resolved mode and port, evidence method, whether process identity was verified, action
result, and key counters. Distinguish `API unreachable`, `log says stopped`, and `process verified
stopped`; they are not interchangeable.
