---
name: control-dino-bot
description: Precisely monitor and safely operate one or more Dino Mutant Bot instances (such as main and S13) from a remote Codex session through the localhost multi-instance Dashboard API, exact instances.json identity, verified Windows process/config/port guards, direct Bot status APIs, and instance-aware shared-log fallback. Use for running/stopped checks, cross-instance device-collision checks, hunting or hatch progress, current stage, recent actions, failures, health, screenshots, diagnostics, start/stop/restart, game restart, single-stage hatch runs, speed profile, WSL/Windows connectivity, or status-port questions. Never expose localhost services externally or use arbitrary ADB, game exploration, raw process killing, or unrequested state changes.
---

# Control Dino Mutant Bot

Treat “remote” as an authenticated Codex session executing on the Bot host. Keep Dashboard and Bot
APIs bound to loopback. Never create a public bind, tunnel, proxy, firewall rule, or port forward.

## Prefer the multi-instance Dashboard

Use the bundled Windows-side wrapper. Convert its exact WSL path with `wslpath -w` and invoke it
through `powershell.exe` so `127.0.0.1` refers to Windows:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File <dashboard-control.ps1> -Action status -Instance all
```

The Dashboard must identify itself as `dino-dashboard` on `127.0.0.1:8780`. It resolves each
instance from the deployed `instances.json`, including its exact config, status port, allowed modes,
ADB serial, logs, and process. Never guess these values from the instance name.

For every multi-instance status check:

1. Query `-Action status -Instance all`, even if the user asks about one side, when collision could
   explain the symptom.
2. Compare running instances' non-empty ADB serials. If two use the same serial, report the
   collision and do not mutate either one until the user chooses what to stop or fixes the mapping.
3. Report the requested instance's port, serial, running state, feature, workflow/current stage,
   last-log time, last action, successful hunts, total actions, verification failures, black-screen
   detections, game restarts, and last successful hunt.
4. Use a second snapshot before calling a run stuck. Slow capture/detection with newer timestamps is
   activity; repeated recovery without workflow advancement is running but not progressing.

Dashboard status is strong read-only evidence but does not independently prove Windows port
ownership. A successful Dashboard stop/restart is stronger: the Dashboard verifies API PID,
Windows port owner, `python.exe`, `main.py`, exact instance config, and status-port arguments before
forwarding control.

## Allowlisted actions

Only perform a state-changing action when explicitly requested in the current turn. Pass `-Confirm`
only after that authorization:

```powershell
... -Action start-hunt       -Instance main -Confirm
... -Action start-hatch-hunt -Instance s13  -Confirm
... -Action start-custom-workflow -Instance s13 -Confirm
... -Action start-stage-hp   -Instance s13  -Confirm
... -Action stop             -Instance s13  -Confirm
... -Action restart-bot      -Instance main -Confirm
... -Action restart-game     -Instance main -Confirm
```

Allowed single stages are `hatch`, `attack`, `hp`, `collect`, and `cave`; the instance's
`allowed_modes` remains authoritative. `scan-adb` is read-only. Run `snapshot` or `diagnostics` only
when requested. After start or restart, query status once and report the Dashboard operation state.

Do not use the legacy `control-windows.ps1` to start a non-default instance: that controller may be
hard-wired to `app/config.json`. Use it only when the resolved instance config is exactly that file.

## Fallback when Dashboard is unavailable

An unreachable Dashboard does not mean a Bot is stopped. For read-only evidence, run the
instance-aware log inspector:

```bash
python3 <skill-root>/scripts/inspect_runtime.py \
  --runtime-root /mnt/d/DinoMutantBot-App --instance main --mode auto
```

Interpret states exactly:

- `active_recently`: log activity within 30 seconds; strong activity evidence, no process identity.
- `stopped_by_log`: the selected instance's last session marker is stop; historical evidence only.
- `unknown`: stale or incomplete evidence.

If Windows interop works and the exact status port is known from `instances.json`, a direct
`control-windows.ps1 -Action status` may supplement read-only evidence. Its status action verifies
the Dino API service but not full process/config ownership. Never use log inference, direct HTTP, or
file edits as a substitute for an authorized mutation.

## Monitoring

For “watch” or “monitor,” take compact Dashboard snapshots periodically and compare timestamps,
counters, workflow stage, and last action. Keep the user updated at least once per minute. Stop
monitoring when the requested terminal condition occurs, the Bot stops, identity becomes ambiguous,
or the user asks to stop. Do not treat unchanged state alone as failure before its configured stall
window expires.

## Hard boundaries

- Never run raw `adb`, `taskkill`, `Stop-Process`, taps, swipes, or arbitrary process commands.
- Never expose ports 8780, 8765, 8775, or any Bot/Dashboard port beyond loopback.
- Never guess runtime roots, instance IDs, ports, configs, serials, package names, or PID identity.
- Never mutate config/source/templates as a substitute for runtime control.
- Stop mutations on missing confirmation, Dashboard/API identity failure, serial collision, unknown
  instance, occupied unverified port, or process/config/port mismatch.
- Port cleanup and any launcher-displayed confirmation token remain human-only.

## Report

State the resolved instance, serial, mode, port, evidence method, whether process ownership was
verified, action result, operation state, and key counters. Distinguish `Dashboard unreachable`,
`Bot API unreachable`, `log says stopped`, and `process verified stopped`.
