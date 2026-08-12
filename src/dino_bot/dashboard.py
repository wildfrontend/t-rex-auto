"""Loopback-only web dashboard for Bot statistics and safe controls."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from .hatch_inventory import HatchBoostInventoryStore
from .metrics import MetricsStore

DASHBOARD_VERSION = 1
DEFAULT_BOT_PORTS = {"hatch-hunt": 8773, "hunt": 8765, "hatch-stage": 8774}
SUPPORTED_BOT_MODES = frozenset(
    {"hunt", "hatch-beginner", "hatch-beginner-hunt", "hatch-hunt", "hatch-stage"}
)
MODE_SWITCH_PROCESS_WAIT_SECONDS = 20
DEFAULT_INSTANCE_ID = "main"
DEFAULT_INSTANCE_NAME = "主力模擬器"
HATCH_STAGE_LABELS = {
    "hatch": "孵蛋一輪",
    "attack": "攻擊親代",
    "hp": "HP 親代",
    "top": "頂尖自動放置",
    "mass": "量產自動放置",
    "collect": "收集所有巢蛋",
    "cave": "洞穴容量與淘汰",
}
_ASSET_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
_LOG_LINE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}) \| (?P<level>[^|]+) \| (?P<message>.*)$"
)
_PLANNING_TARGET = re.compile(r"^Planning \| (?P<target>\S+) at \(")
_COLLECT_TARGETS = frozenset(
    {
        "hatch_collect_eggs_button",
        "hatch_nest_mask_close",
    }
)
_CAVE_TARGETS = frozenset(
    {
        "hatch_cave_swipe",
        "hatch_cave_recenter",
        "cave",
        "hatch_cave_select_button",
        "hatch_cull_tag_header",
        "hatch_select_weakest_button",
        "hatch_select_choose_button",
        "hatch_cave_continuous_button",
        "hatch_cave_close_button",
    }
)


@dataclass(frozen=True, slots=True)
class BotInstance:
    """A separately addressable Bot configuration managed by the dashboard."""

    instance_id: str
    name: str
    config_path: Path
    status_port: int
    allowed_modes: frozenset[str] = SUPPORTED_BOT_MODES

    @property
    def root(self) -> Path:
        return self.config_path.parent

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def database(self) -> Path:
        return self.root / "data" / "stats.sqlite3"


def _loopback_origin_allowed(origin: str) -> bool:
    if not origin:
        return True
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "::1",
        "localhost",
    }


def _get_json(url: str, timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        with urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback URL only
            return json.load(response)
    except (OSError, ValueError, URLError):
        return None


def _post_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    request = Request(url, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback URL only
            return json.load(response)
    except HTTPError as exc:
        try:
            payload = json.load(exc)
        except (OSError, ValueError):
            payload = {}
        reason = payload.get("error") if isinstance(payload, dict) else None
        raise RuntimeError(f"Bot 拒絕控制指令：{reason or exc.reason}") from exc
    except (OSError, ValueError, URLError) as exc:
        raise RuntimeError(f"Bot control unavailable: {exc}") from exc


def _port_is_bindable(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def _process_exists(process_id: int | None) -> bool:
    if process_id is None:
        return False
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _windows_port_owner(port: int) -> dict[str, Any] | None:
    script = (
        f"$c=Get-NetTCPConnection -LocalPort {int(port)} -State Listen "
        "-ErrorAction SilentlyContinue | Select-Object -First 1;"
        "if($null -ne $c){"
        "$p=Get-CimInstance Win32_Process "
        "-Filter ('ProcessId='+$c.OwningProcess) -ErrorAction SilentlyContinue;"
        "[pscustomobject]@{pid=$c.OwningProcess;name=$p.Name;command=$p.CommandLine}"
        "| ConvertTo-Json -Compress}"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        owner = json.loads(completed.stdout)
    except (TypeError, ValueError):
        return None
    return owner if isinstance(owner, dict) else None


def _port_occupant_detail(port: int) -> str:
    """Return read-only owner detail where the host can provide it."""
    if os.name == "nt":
        owner = _windows_port_owner(port)
        if owner is not None and owner.get("pid") is not None:
            return (
                f"PID {owner['pid']}（{owner.get('name') or 'unknown'} / "
                f"{owner.get('command') or '命令列無法讀取'}）"
            )
        return "Windows 無法取得占用程序的 PID 或命令列"
    lsof = Path("/usr/sbin/lsof")
    executable = str(lsof) if lsof.is_file() else shutil.which("lsof")
    if not executable:
        return "無法從目前系統取得占用程序的 PID"
    completed = subprocess.run(
        [executable, "-nP", "-a", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = sorted(
        {
            int(line[1:])
            for line in completed.stdout.splitlines()
            if line.startswith("p") and line[1:].isdigit()
        }
    )
    if not pids:
        return "沒有可辨識的 LISTEN PID，但 port 目前仍無法綁定"
    owners: list[str] = []
    for pid in pids:
        command = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        owners.append(f"PID {pid}（{command or '命令列無法讀取'}）")
    return "、".join(owners)


def _latest_log_lines(logs_dir: Path, limit: int = 4000) -> list[str]:
    # 狩獵每次貢獻約 6 行 INFO;視窗太小會把冷卻等待行擠出去,
    # 進度條就退回預設的「檢查孵蛋」。4000 行足以涵蓋最長冷卻。
    paths = sorted(logs_dir.glob("20*.log"))
    if not paths:
        return []
    try:
        with paths[-1].open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - 2 * 1024 * 1024))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = [line for line in text.splitlines() if " | DEBUG | " not in line]
    return lines[-max(1, limit) :]


def _workflow_status(logs_dir: Path, mode: str | None) -> dict[str, Any]:
    stage = "hunt" if mode == "hunt" else "hatch"
    label = "純狩獵" if mode == "hunt" else "檢查孵蛋"
    cooldown_remaining: int | None = None
    latest_date: str | None = None
    paths = sorted(logs_dir.glob("20*.log"))
    if paths:
        try:
            latest_date = datetime.strptime(paths[-1].stem, "%Y%m%d").date().isoformat()
        except ValueError:
            latest_date = None

    for line in _latest_log_lines(logs_dir):
        match = _LOG_LINE.match(line)
        if match is None:
            continue
        time_text = match.group("time")
        message = match.group("message")
        planning = _PLANNING_TARGET.match(message)
        planned_target = planning.group("target") if planning is not None else None
        if message.startswith("Feature | hatch-stage | stage="):
            stage_name = message.split("stage=", 1)[1].split(" ", 1)[0]
            stage = {
                "hatch": "hatch",
                "attack": "nest_attack",
                "hp": "nest_hp",
                "top": "nest_top",
                "mass": "nest_mass",
                "collect": "collect",
                "cave": "cave",
            }.get(stage_name, "hatch")
            label = HATCH_STAGE_LABELS.get(stage_name, "單階段孵化")
        elif "Hatch full | screening stage starting | stage=" in message:
            stage_name = message.split("stage=", 1)[1].split(" ", 1)[0]
            stage, label = {
                "attack": ("nest_attack", "篩選攻擊親代"),
                "hp": ("nest_hp", "篩選 HP 親代"),
                "top": ("nest_top", "頂尖自動放置"),
                "mass": ("nest_mass", "量產自動放置"),
            }.get(stage_name, ("nest_attack", "篩選攻擊親代"))
        elif "Hatch full | screening complete" in message:
            stage, label = "collect", "收集所有巢蛋"
        elif "Hatch full | phase A complete" in message:
            # 只有觸發篩選的批次才會進入親代篩選;一般循環接著收蛋。
            if "trigger=none" in message:
                stage, label = "collect", "收集所有巢蛋"
            else:
                stage, label = "nest_attack", "篩選攻擊親代"
        elif "Hatch 攻擊特化 |" in message:
            stage, label = "nest_attack", "篩選攻擊親代"
        elif "Hatch HP特化 |" in message:
            stage, label = "nest_hp", "篩選 HP 親代"
        elif "Hatch auto-place | tag=頂尖" in message:
            stage, label = "nest_top", "頂尖自動放置"
        elif "Hatch auto-place | tag=量產" in message:
            stage, label = "nest_mass", "量產自動放置"
        elif (
            "Hatch full | no ready incubator eggs" in message
            or planned_target in _COLLECT_TARGETS
        ):
            stage, label = "collect", "收集所有巢蛋"
        elif "Hatch cave |" in message or planned_target in _CAVE_TARGETS:
            stage, label = "cave", "洞穴容量與淘汰"
        elif "Hatch full | completed management cycle" in message:
            stage, label = "hatch", "檢查孵蛋"
        elif "Hatch+Hunt | cooldown handoff window" in message:
            stage, label = "handoff", "返回孵蛋首頁"
        elif "Hatch+Hunt | centered home confirmed" in message:
            stage, label = "hatch", "檢查孵蛋"
            cooldown_remaining = 0
        elif "Hatch+Hunt | egg cooldown" in message and "switching to hunt" in message:
            stage, label = "cooldown_hunt", "冷卻期間狩獵"
        elif "interim nest collection during hunt idle" in message:
            # 差事出發時帶著剩餘冷卻;收完會自動回狩獵。
            try:
                duration = int(float(message.split("resume=", 1)[1].split("s", 1)[0]))
                if latest_date:
                    started = datetime.fromisoformat(f"{latest_date}T{time_text}").astimezone()
                    elapsed = (datetime.now().astimezone() - started).total_seconds()
                    cooldown_remaining = max(0, round(duration - elapsed))
                    stage, label = "cooldown_hunt", "冷卻期間狩獵"
            except (ValueError, IndexError):
                pass
        elif "Hatch | wait " in message:
            # 兩種等待都算冷卻:關閉孵化器後、收完巢蛋後。
            try:
                duration = int(float(message.split("Hatch | wait ", 1)[1].split("s", 1)[0]))
                if latest_date:
                    started = datetime.fromisoformat(f"{latest_date}T{time_text}").astimezone()
                    elapsed = (datetime.now().astimezone() - started).total_seconds()
                    cooldown_remaining = max(0, round(duration - elapsed))
                    if cooldown_remaining > 30:
                        stage, label = "cooldown_hunt", "冷卻期間狩獵"
                    elif cooldown_remaining > 0:
                        stage, label = "handoff", "返回孵蛋首頁"
                    else:
                        # 冷卻已過:回孵化器檢查,不再殘留舊狀態。
                        stage, label = "hatch", "檢查孵蛋"
            except (ValueError, IndexError):
                pass
    return {
        "stage": stage,
        "label": label,
        "cooldown_remaining_seconds": cooldown_remaining,
    }


class DashboardController:
    """Discover and control multiple independently configured Bot instances."""

    def __init__(
        self,
        runtime_root: Path,
        logs_dir: Path,
        *,
        bot_ports: dict[str, int] | None = None,
        config_path: Path | None = None,
        instances_path: Path | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        candidate = runtime_root / "app"
        self.app_root = candidate if candidate.is_dir() else runtime_root
        self.logs_dir = logs_dir
        self.config_path = (config_path or self.app_root / "config.json").resolve()
        self.bot_ports = dict(bot_ports or DEFAULT_BOT_PORTS)
        self.instances_path = (instances_path or runtime_root / "instances.json").resolve()
        self._instances = self._load_instances()
        self._start_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._last_start: dict[str, float] = {}
        self._explicit_stop_at: dict[str, float] = {}
        self._operations: dict[str, dict[str, Any]] = {}

    @property
    def instances(self) -> tuple[BotInstance, ...]:
        return tuple(self._instances)

    def _load_instances(self) -> list[BotInstance]:
        fallback = BotInstance(
            DEFAULT_INSTANCE_ID,
            DEFAULT_INSTANCE_NAME,
            self.config_path,
            DEFAULT_BOT_PORTS["hunt"],
        )
        if not self.instances_path.is_file():
            return [fallback]
        try:
            payload = json.loads(self.instances_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return [fallback]
        raw_instances = payload.get("instances") if isinstance(payload, dict) else None
        if not isinstance(raw_instances, list):
            return [fallback]
        instances: list[BotInstance] = []
        seen_ids: set[str] = set()
        seen_ports: set[int] = set()
        for raw in raw_instances:
            if not isinstance(raw, dict):
                continue
            instance_id = str(raw.get("id", "")).strip()
            name = str(raw.get("name", instance_id)).strip() or instance_id
            raw_config = raw.get("config")
            if not instance_id or not isinstance(raw_config, str):
                continue
            config_path = Path(raw_config).expanduser()
            if not config_path.is_absolute():
                config_path = self.instances_path.parent / config_path
            try:
                status_port = int(raw.get("status_port", 0))
            except (TypeError, ValueError):
                continue
            raw_modes = raw.get("allowed_modes")
            if raw_modes is None:
                allowed_modes = SUPPORTED_BOT_MODES
            elif isinstance(raw_modes, list):
                allowed_modes = frozenset(str(mode) for mode in raw_modes)
            else:
                continue
            if (
                instance_id in seen_ids
                or not 1 <= status_port <= 65535
                or status_port in seen_ports
                or not config_path.is_file()
                or not allowed_modes
                or not allowed_modes <= SUPPORTED_BOT_MODES
            ):
                continue
            seen_ids.add(instance_id)
            seen_ports.add(status_port)
            instances.append(
                BotInstance(
                    instance_id,
                    name,
                    config_path.resolve(),
                    status_port,
                    allowed_modes,
                )
            )
        return instances or [fallback]

    def _save_instances(self) -> None:
        def config_reference(instance: BotInstance) -> str:
            try:
                return str(instance.config_path.relative_to(self.instances_path.parent))
            except ValueError:
                return str(instance.config_path)

        payload = {
            "instances": [
                {
                    "id": instance.instance_id,
                    "name": instance.name,
                    "config": config_reference(instance),
                    "status_port": instance.status_port,
                    "allowed_modes": sorted(instance.allowed_modes),
                }
                for instance in self._instances
            ]
        }
        self.instances_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.instances_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.instances_path)

    def _instance(self, instance_id: str | None = None) -> BotInstance:
        if instance_id is None:
            return self._instances[0]
        for instance in self._instances:
            if instance.instance_id == instance_id:
                return instance
        raise RuntimeError(f"Unknown Bot instance: {instance_id}")

    def scan_adb(self, instance_id: str | None = None) -> dict[str, Any]:
        """Probe the known emulator ports on behalf of one instance.

        The port is the one setting a user cannot read off the emulator's own
        UI, so leaving it as free text with no way to check is what turns a
        typo into "the bot does nothing".
        """

        from .actions import AdbClient
        from .config import ConfigError, load_config

        instance = self._instance(instance_id)
        try:
            config = load_config(instance.config_path)
        except ConfigError as exc:
            raise RuntimeError(f"設定檔讀取失敗:{exc}") from exc
        client = AdbClient(config.adb)
        devices = [device.as_dict() for device in client.discover()]
        ready = [device for device in devices if device["ready"]]
        if not devices:
            message = "沒有找到任何裝置。請確認模擬器已啟動,且設定裡開啟了 ADB。"
        elif len(ready) == 1:
            message = f"找到 1 台:{ready[0]['serial']}"
        else:
            message = f"找到 {len(devices)} 台,請選擇要使用的那一台。"
        return {
            "accepted": True,
            "action": "scan-adb",
            "instance_id": instance.instance_id,
            "configured_serial": self._instance_serial(instance),
            "devices": devices,
            "message": message,
        }

    @staticmethod
    def _instance_serial(instance: BotInstance) -> str | None:
        try:
            payload = json.loads(instance.config_path.read_text(encoding="utf-8"))
            adb = payload.get("adb", {})
            serial = adb.get("serial") if isinstance(adb, dict) else None
            return str(serial) if serial else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _mode_info(feature: str | None) -> tuple[str | None, str]:
        if feature == "hunt":
            return "hunt", "純狩獵"
        if feature == "hatch-beginner":
            return "hatch-beginner", "新手自動孵蛋"
        if feature == "hatch-beginner-hunt":
            return "hatch-beginner-hunt", "新手孵蛋＋狩獵"
        if feature == "hatch-hunt":
            return "hatch-hunt", "自動孵蛋＋狩獵"
        if feature and feature.startswith("hatch-stage-"):
            return "hatch-stage", "單階段孵化"
        return None, "執行中"

    @staticmethod
    def _expected_feature(mode: str, stage: str | None = None) -> str:
        if mode == "hatch-stage" and stage:
            return f"hatch-stage-{stage}"
        return mode

    def _set_operation(
        self,
        instance: BotInstance,
        *,
        action: str,
        state: str,
        code: str,
        message: str,
    ) -> dict[str, Any]:
        operation = {
            "action": action,
            "state": state,
            "code": code,
            "message": message,
            "updated_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        }
        with self._operation_lock:
            self._operations[instance.instance_id] = operation
        return dict(operation)

    def operation(self, instance_id: str | None = None) -> dict[str, Any] | None:
        instance = self._instance(instance_id)
        with self._operation_lock:
            operation = self._operations.get(instance.instance_id)
            return dict(operation) if operation is not None else None

    @staticmethod
    def _launch_log_tail(instance: BotInstance, mode: str, lines: int = 10) -> str:
        failure_file = instance.logs_dir / "dashboard-launch-failure.json"
        try:
            failure = json.loads(failure_file.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            failure = None
        if isinstance(failure, dict) and failure.get("message"):
            return f"Windows runner：{failure['message']}"
        path = instance.logs_dir / f"dashboard-launch-{mode}.log"
        try:
            entries = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "啟動 log 不存在或無法讀取"
        tail = [entry.strip() for entry in entries[-lines:] if entry.strip()]
        return " | ".join(tail) if tail else "啟動 log 沒有內容"

    @staticmethod
    def _port_block_reason(instance: BotInstance) -> str:
        health = _get_json(f"http://127.0.0.1:{instance.status_port}/health")
        if health and health.get("service") == "dino-mutant-bot-status":
            pid = health.get("process_id", "未知")
            feature = health.get("feature", "未知模式")
            return (
                f"Port {instance.status_port} 仍由 Dino Bot PID {pid} 使用"
                f"（feature={feature}），舊程序尚未完成退出"
            )
        if health:
            return (
                f"Port {instance.status_port} 有 API 回應，但 service={health.get('service')!r}，"
                "不是 Dino Bot status API"
            )
        return f"Port {instance.status_port} 被占用：{_port_occupant_detail(instance.status_port)}"

    def _require_clean_port(self, instance: BotInstance) -> None:
        if not _port_is_bindable(instance.status_port):
            raise RuntimeError(self._port_block_reason(instance))

    def _wait_for_clean_port(
        self,
        instance: BotInstance,
        expected_process_id: int | None,
        timeout_seconds: float = MODE_SWITCH_PROCESS_WAIT_SECONDS,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if (
                not _process_exists(expected_process_id)
                and _port_is_bindable(instance.status_port)
            ):
                return
            time.sleep(0.25)
        if _process_exists(expected_process_id):
            raise RuntimeError(
                f"等待 {timeout_seconds:g} 秒後仍不能乾淨重啟：舊 Bot PID "
                f"{expected_process_id} 尚未退出；{self._port_block_reason(instance)}"
            )
        raise RuntimeError(
            f"等待 {timeout_seconds:g} 秒後仍不能乾淨重啟："
            f"{self._port_block_reason(instance)}"
        )

    def _wait_for_ready(
        self,
        instance: BotInstance,
        mode: str,
        stage: str | None,
        launcher_process: subprocess.Popen[Any] | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        expected_feature = self._expected_feature(mode, stage)
        deadline = time.monotonic() + timeout_seconds
        last_reason = "新程序尚未監聽 status port"
        while time.monotonic() < deadline:
            if launcher_process is not None and launcher_process.poll() is not None:
                raise RuntimeError(
                    f"Bot runner PID {launcher_process.pid} 在 API 就緒前退出"
                    f"（exit code {launcher_process.returncode}）。"
                    f"原因：{self._launch_log_tail(instance, mode)}"
                )
            health = _get_json(f"http://127.0.0.1:{instance.status_port}/health")
            if health:
                if health.get("service") != "dino-mutant-bot-status":
                    last_reason = f"service={health.get('service')!r}，不是 Dino Bot status API"
                elif health.get("feature") != expected_feature:
                    last_reason = (
                        f"API 模式不符：預期 {expected_feature}，"
                        f"實際 {health.get('feature')!r}"
                    )
                else:
                    status = _get_json(
                        f"http://127.0.0.1:{instance.status_port}/status"
                    )
                    if status and status.get("running") is not False:
                        return
                    last_reason = "status API 已回應，但 running=false 或狀態無法讀取"
            elif not _port_is_bindable(instance.status_port):
                last_reason = self._port_block_reason(instance)
            time.sleep(0.25)
        raise RuntimeError(
            f"新 Bot 在 {timeout_seconds:g} 秒內未就緒：{last_reason}。"
            f"最近啟動 log：{self._launch_log_tail(instance, mode)}"
        )

    def _monitor_started(
        self,
        instance: BotInstance,
        mode: str,
        stage: str | None,
        action: str,
        launcher_process: subprocess.Popen[Any] | None = None,
    ) -> None:
        try:
            self._wait_for_ready(
                instance,
                mode,
                stage,
                launcher_process=launcher_process,
            )
        except RuntimeError as exc:
            self._set_operation(
                instance,
                action=action,
                state="failed",
                code="start_failed",
                message=str(exc),
            )
            return
        self._set_operation(
            instance,
            action=action,
            state="succeeded",
            code="bot_ready",
            message=(
                f"Bot 已完成啟動驗證：{self._expected_feature(mode, stage)}，"
                f"Port {instance.status_port}"
            ),
        )

    def _discover_instance(self, instance: BotInstance) -> dict[str, Any]:
        health = _get_json(f"http://127.0.0.1:{instance.status_port}/health")
        status = _get_json(f"http://127.0.0.1:{instance.status_port}/status") or {}
        if health and health.get("service") == "dino-mutant-bot-status":
            mode, mode_label = self._mode_info(health.get("feature"))
            return {
                "instance_id": instance.instance_id,
                "name": instance.name,
                "serial": self._instance_serial(instance),
                "running": bool(status.get("running", True)),
                "mode": mode,
                "feature": health.get("feature"),
                "process_id": health.get("process_id"),
                "mode_label": mode_label,
                "port": instance.status_port,
                "status": status,
                "workflow": _workflow_status(instance.logs_dir, mode),
            }
        return {
            "instance_id": instance.instance_id,
            "name": instance.name,
            "serial": self._instance_serial(instance),
            "running": False,
            "mode": None,
            "feature": None,
            "process_id": None,
            "mode_label": "未啟動",
            "port": instance.status_port,
            "status": {},
            "workflow": _workflow_status(instance.logs_dir, None),
        }

    def discover(self, instance_id: str | None = None) -> dict[str, Any]:
        if instance_id is not None:
            return self._discover_instance(self._instance(instance_id))
        discovered = [self._discover_instance(instance) for instance in self._instances]
        for active in discovered:
            if active["running"]:
                return active
        return discovered[0]

    def discover_all(self) -> list[dict[str, Any]]:
        return [self._discover_instance(instance) for instance in self._instances]

    def _bot_command(
        self,
        mode: str,
        *,
        stage: str | None = None,
        instance_id: str | None = None,
    ) -> list[str]:
        """Direct Python launch used on non-Windows hosts (no PowerShell runner)."""

        instance = self._instance(instance_id)
        command = [
            sys.executable,
            str(self.app_root / "main.py"),
            "--config",
            str(instance.config_path),
        ]
        if mode == "hunt":
            command += [
                "run",
                "--mode",
                "runtime",
                "--max-actions",
                "0",
                "--max-cycles",
                "0",
                "--batch-size",
                "10",
                "--mail-after-hunts",
                "30",
                "--speed",
                "fast",
                "--status-port",
                str(instance.status_port),
                "--verbose",
            ]
        elif mode in {"hatch-beginner", "hatch-beginner-hunt", "hatch-hunt"}:
            command += [
                "run",
                "--feature",
                mode,
                "--mode",
                "debug",
                "--speed",
                "safe",
                "--status-port",
                str(instance.status_port),
                "--max-actions",
                "0",
                "--max-cycles",
                "0",
                "--verbose",
            ]
        elif mode == "hatch-stage" and stage in HATCH_STAGE_LABELS:
            command += [
                "run",
                "--feature",
                f"hatch-stage-{stage}",
                "--mode",
                "debug",
                "--speed",
                "safe",
                "--status-port",
                str(instance.status_port),
                "--max-actions",
                "0",
                "--max-cycles",
                "0",
                "--verbose",
            ]
        else:
            raise RuntimeError("Unsupported Bot mode")
        return command

    def _runner_command(
        self,
        mode: str,
        *,
        stage: str | None = None,
        instance_id: str | None = None,
        wait_for_existing_seconds: int = 0,
    ) -> list[str]:
        instance = self._instance(instance_id)
        scripts = self.app_root / "scripts"
        if mode == "hunt":
            runner = scripts / "run-windows.ps1"
            arguments = [
                "-Mode",
                "runtime",
                "-MaxActions",
                "0",
                "-MaxCycles",
                "0",
                "-BatchSize",
                "10",
                "-MailAfterHunts",
                "30",
                "-ConfigPath",
                str(instance.config_path),
                "-Speed",
                "fast",
                "-StatusPort",
                str(instance.status_port),
            ]
        elif mode in {"hatch-beginner", "hatch-beginner-hunt", "hatch-hunt"}:
            runner = scripts / "run-hatch-windows.ps1"
            arguments = [
                "-Feature",
                mode,
                "-Mode",
                "debug",
                "-ConfigPath",
                str(instance.config_path),
                "-Speed",
                "safe",
                "-StatusPort",
                str(instance.status_port),
                "-MaxActions",
                "0",
                "-MaxCycles",
                "0",
            ]
        elif mode == "hatch-stage" and stage in HATCH_STAGE_LABELS:
            runner = scripts / "run-hatch-windows.ps1"
            arguments = [
                "-Feature",
                f"hatch-stage-{stage}",
                "-Mode",
                "debug",
                "-ConfigPath",
                str(instance.config_path),
                "-Speed",
                "safe",
                "-StatusPort",
                str(instance.status_port),
                "-MaxActions",
                "0",
                "-MaxCycles",
                "0",
            ]
        else:
            raise RuntimeError("Unsupported Bot mode")
        if wait_for_existing_seconds > 0:
            arguments += [
                "-WaitForExistingSeconds",
                str(wait_for_existing_seconds),
            ]
        runner = runner.resolve()
        if runner.parent != scripts.resolve() or not runner.is_file():
            raise RuntimeError(f"Runner not found: {runner}")
        return [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runner),
            *arguments,
        ]

    def start(
        self,
        mode: str,
        *,
        stage: str | None = None,
        instance_id: str | None = None,
    ) -> dict[str, Any]:
        instance = self._instance(instance_id)
        if mode not in {
            "hunt",
            "hatch-beginner",
            "hatch-beginner-hunt",
            "hatch-hunt",
            "hatch-stage",
        }:
            raise RuntimeError("Unsupported Bot mode")
        if mode not in instance.allowed_modes:
            raise RuntimeError(f"{instance.name} 不允許啟動 {mode}")
        if mode == "hatch-stage" and stage not in HATCH_STAGE_LABELS:
            raise RuntimeError("Unsupported hatch stage")
        with self._start_lock:
            now = time.monotonic()
            if now - self._last_start.get(instance.instance_id, 0) < 5:
                operation = self.operation(instance.instance_id)
                detail = operation.get("message") if operation else "尚無詳細狀態"
                raise RuntimeError(f"Bot 啟動要求仍在處理中：{detail}")
            active = self.discover(instance.instance_id)
            if not active["running"]:
                self._require_clean_port(instance)
                self._last_start[instance.instance_id] = now
                operation = self._set_operation(
                    instance,
                    action="start",
                    state="pending",
                    code="launching",
                    message=(
                        f"正在啟動 {self._expected_feature(mode, stage)}；"
                        f"等待 Port {instance.status_port} status API 驗證"
                    ),
                )
                try:
                    launcher_process = self._launch(
                        mode,
                        stage=stage,
                        instance_id=instance.instance_id,
                    )
                except RuntimeError as exc:
                    self._set_operation(
                        instance,
                        action="start",
                        state="failed",
                        code="launcher_create_failed",
                        message=str(exc),
                    )
                    raise
                threading.Thread(
                    target=self._monitor_started,
                    args=(instance, mode, stage, "start", launcher_process),
                    daemon=True,
                ).start()
                return {
                    "accepted": True,
                    "action": "start",
                    "instance_id": instance.instance_id,
                    "mode": mode,
                    "stage": stage,
                    "message": operation["message"],
                    "operation": operation,
                }
            previous_label = str(active["mode_label"])
            previous_process_id = active.get("process_id")
            self.stop(instance.instance_id)
            self._last_start[instance.instance_id] = now
            operation = self._set_operation(
                instance,
                action="switch",
                state="pending",
                code="stopping_old_bot",
                message=(
                    f"正在停止{previous_label}；確認舊程序退出及 Port "
                    f"{instance.status_port} 釋放後，將啟動 {self._expected_feature(mode, stage)}"
                ),
            )

        def start_after_stop() -> None:
            try:
                self._wait_for_clean_port(instance, previous_process_id)
                with self._start_lock:
                    self._require_clean_port(instance)
                    launcher_process = self._launch(
                        mode,
                        stage=stage,
                        instance_id=instance.instance_id,
                        wait_for_existing_seconds=MODE_SWITCH_PROCESS_WAIT_SECONDS,
                    )
                self._monitor_started(
                    instance,
                    mode,
                    stage,
                    "switch",
                    launcher_process,
                )
            except RuntimeError as exc:
                self._set_operation(
                    instance,
                    action="switch",
                    state="failed",
                    code="clean_restart_failed",
                    message=str(exc),
                )

        threading.Thread(target=start_after_stop, daemon=True).start()
        return {
            "accepted": True,
            "action": "switch",
            "instance_id": instance.instance_id,
            "mode": mode,
            "stage": stage,
            "message": operation["message"],
            "operation": operation,
        }

    def _launch(
        self,
        mode: str,
        *,
        stage: str | None = None,
        instance_id: str | None = None,
        wait_for_existing_seconds: int = 0,
    ) -> subprocess.Popen[Any]:
        instance = self._instance(instance_id)
        try:
            if os.name == "nt":
                creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) | getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
                failure_file = instance.logs_dir / "dashboard-launch-failure.json"
                instance.logs_dir.mkdir(parents=True, exist_ok=True)
                with suppress(OSError):
                    failure_file.unlink()
                command = self._runner_command(
                    mode,
                    stage=stage,
                    instance_id=instance.instance_id,
                    wait_for_existing_seconds=wait_for_existing_seconds,
                )
                command += ["-FailureFile", str(failure_file)]
                return subprocess.Popen(  # noqa: S603 - fixed local runner and allowlist
                    command,
                    cwd=self.runtime_root,
                    creationflags=creation_flags,
                )
            instance.logs_dir.mkdir(parents=True, exist_ok=True)
            launch_log = instance.logs_dir / f"dashboard-launch-{mode}.log"
            with launch_log.open("ab") as stream:
                return subprocess.Popen(  # noqa: S603 - fixed local entrypoint and allowlist
                    self._bot_command(mode, stage=stage, instance_id=instance.instance_id),
                    cwd=instance.root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            raise RuntimeError(
                f"無法建立 Bot runner 程序：{type(exc).__name__}: {exc}"
            ) from exc

    def _active_control(
        self,
        action: str,
        instance_id: str | None = None,
    ) -> dict[str, Any]:
        instance = self._instance(instance_id)
        active = self.discover(instance.instance_id)
        if not active["running"] or active["port"] is None:
            raise RuntimeError(f"Bot instance '{instance.name}' is not running")
        if os.name == "nt":
            owner = _windows_port_owner(instance.status_port)
            command = str((owner or {}).get("command") or "")
            owner_pid = (owner or {}).get("pid")
            api_pid = active.get("process_id")
            main_script = str((self.app_root / "main.py").resolve())
            port_pattern = rf"--status-port\s+{instance.status_port}(?:\s|$)"
            if (
                owner is None
                or owner_pid != api_pid
                or str((owner or {}).get("name", "")).lower() != "python.exe"
                or main_script.lower() not in command.lower()
                or str(instance.config_path).lower() not in command.lower()
                or " run " not in command
                or re.search(port_pattern, command, re.IGNORECASE) is None
            ):
                raise RuntimeError(
                    f"拒絕控制 Port {instance.status_port}：API PID={api_pid}，"
                    f"實際占用程序={_port_occupant_detail(instance.status_port)}；"
                    "PID、main.py、config 或 status-port 身分不一致"
                )
        response = _post_json(f"http://127.0.0.1:{active['port']}/control/{action}")
        if response.get("accepted") is not True or response.get("action") != action:
            raise RuntimeError(
                f"Bot 未接受 {action} 指令：{json.dumps(response, ensure_ascii=False)}"
            )
        return response

    def stop(self, instance_id: str | None = None) -> dict[str, Any]:
        instance = self._instance(instance_id)
        self._explicit_stop_at[instance.instance_id] = time.monotonic()
        return self._active_control("stop", instance.instance_id)

    def restart_game(self, instance_id: str | None = None) -> dict[str, Any]:
        return self._active_control("restart-game", instance_id)

    def restart_bot(self, instance_id: str | None = None) -> dict[str, Any]:
        instance = self._instance(instance_id)
        active = self.discover(instance.instance_id)
        if not active["running"] or not active["mode"]:
            raise RuntimeError(f"Bot instance '{instance.name}' is not running")
        mode = str(active["mode"])
        previous_process_id = active.get("process_id")
        if mode == "hatch-stage":
            raise RuntimeError("單階段工作不支援重啟；請停止後重新選擇階段")
        self.stop(instance.instance_id)
        self._last_start[instance.instance_id] = time.monotonic()
        operation = self._set_operation(
            instance,
            action="restart",
            state="pending",
            code="stopping_old_bot",
            message=(
                f"正在乾淨重啟：等待舊 PID 退出與 Port {instance.status_port} 釋放，"
                f"再重新啟動 {mode}"
            ),
        )

        def restart_after_stop() -> None:
            try:
                self._wait_for_clean_port(instance, previous_process_id)
                with self._start_lock:
                    self._require_clean_port(instance)
                    launcher_process = self._launch(
                        mode,
                        instance_id=instance.instance_id,
                        wait_for_existing_seconds=MODE_SWITCH_PROCESS_WAIT_SECONDS,
                    )
                self._monitor_started(
                    instance,
                    mode,
                    None,
                    "restart",
                    launcher_process,
                )
            except RuntimeError as exc:
                self._set_operation(
                    instance,
                    action="restart",
                    state="failed",
                    code="clean_restart_failed",
                    message=str(exc),
                )

        threading.Thread(target=restart_after_stop, daemon=True).start()
        return {
            "accepted": True,
            "action": "restart",
            "instance_id": instance.instance_id,
            "mode": mode,
            "message": operation["message"],
            "operation": operation,
        }

    def run_tool(self, action: str, instance_id: str | None = None) -> dict[str, Any]:
        instance = self._instance(instance_id)
        main_script = self.app_root / "main.py"
        config = instance.config_path
        if action == "open-logs":
            instance.logs_dir.mkdir(parents=True, exist_ok=True)
            startfile = getattr(os, "startfile", None)
            if callable(startfile):
                startfile(str(instance.logs_dir))
            elif sys.platform == "darwin":
                subprocess.run(["/usr/bin/open", str(instance.logs_dir)], check=False)
            else:
                raise RuntimeError("Opening folders is only available on Windows and macOS")
            return {"accepted": True, "action": action}
        if action == "diagnostics":
            command = [sys.executable, str(main_script), "--config", str(config), "diagnostics"]
        elif action == "snapshot":
            output = self.app_root / "debug" / (
                f"dashboard-{instance.instance_id}-"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
                + ".png"
            )
            command = [
                sys.executable,
                str(main_script),
                "--config",
                str(config),
                "snapshot",
                "--backend",
                "adb",
                "--output",
                str(output),
            ]
        else:
            raise RuntimeError("Unsupported dashboard tool")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"{action} failed")
        return {
            "accepted": True,
            "action": action,
            "message": completed.stdout.strip(),
        }

    def add_instance(
        self,
        *,
        name: str,
        serial: str,
        status_port: int,
        instance_id: str | None = None,
    ) -> dict[str, Any]:
        """Create an isolated config/assets/log directory for a new emulator."""

        name = str(name).strip()
        serial = str(serial).strip()
        if not name or not serial:
            raise ValueError("instance name and ADB serial are required")
        if not 1 <= int(status_port) <= 65535:
            raise ValueError("status port must be between 1 and 65535")
        if any(instance.status_port == int(status_port) for instance in self._instances):
            raise ValueError(f"status port {status_port} is already assigned")
        if instance_id is None:
            instance_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-_").lower()
            if not instance_id:
                instance_id = f"instance-{len(self._instances) + 1}"
        if not instance_id or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,31}", instance_id):
            raise ValueError("instance id must use 1-32 letters, numbers, '_' or '-'")
        if any(instance.instance_id == instance_id for instance in self._instances):
            raise ValueError(f"instance id '{instance_id}' already exists")

        instance_root = (self.runtime_root / "instances" / instance_id).resolve()
        if instance_root.parent != (self.runtime_root / "instances").resolve():
            raise ValueError("invalid instance path")
        if instance_root.exists():
            raise ValueError(f"instance directory already exists: {instance_id}")
        source = self._instances[0].config_path
        try:
            config = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot read base config: {exc}") from exc
        if not isinstance(config, dict):
            raise RuntimeError("base config must be a JSON object")
        adb = config.setdefault("adb", {})
        if not isinstance(adb, dict):
            raise RuntimeError("base config adb section must be an object")
        adb["serial"] = serial

        instance_root.mkdir(parents=True)
        try:
            shutil.copytree(self.app_root / "assets", instance_root / "assets")
            config_path = instance_root / "config.json"
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            instance = BotInstance(instance_id, name, config_path, int(status_port))
            self._instances.append(instance)
            self._save_instances()
        except Exception:
            shutil.rmtree(instance_root, ignore_errors=True)
            raise
        return {
            "instance_id": instance.instance_id,
            "name": instance.name,
            "serial": serial,
            "status_port": instance.status_port,
            "config": str(instance.config_path),
        }

    def update_instance(
        self,
        *,
        instance_id: str,
        name: str,
        serial: str,
        status_port: int,
        restart: bool = True,
    ) -> dict[str, Any]:
        """Persist instance settings and optionally restart it on the new port."""

        instance = self._instance(instance_id)
        name = str(name).strip()
        serial = str(serial).strip()
        try:
            status_port = int(status_port)
        except (TypeError, ValueError) as exc:
            raise ValueError("status port must be an integer") from exc
        if not name or not serial:
            raise ValueError("instance name and ADB serial are required")
        if len(name) > 40:
            raise ValueError("instance name must be 40 characters or fewer")
        if not 1 <= status_port <= 65535:
            raise ValueError("status port must be between 1 and 65535")
        if any(
            other.instance_id != instance_id and other.status_port == status_port
            for other in self._instances
        ):
            raise ValueError(f"status port {status_port} is already assigned")
        if not isinstance(restart, bool):
            raise ValueError("restart must be a boolean")

        active = self.discover(instance_id)
        was_running = bool(active["running"])
        feature = active.get("feature")
        if was_running:
            self.stop(instance_id)
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if _get_json(f"http://127.0.0.1:{instance.status_port}/health") is None:
                    break
                time.sleep(0.25)

        try:
            config = json.loads(instance.config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot read instance config: {exc}") from exc
        if not isinstance(config, dict):
            raise RuntimeError("instance config must be a JSON object")
        adb = config.setdefault("adb", {})
        if not isinstance(adb, dict):
            raise RuntimeError("instance config adb section must be an object")
        adb["serial"] = serial
        temporary = instance.config_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(instance.config_path)

        updated = BotInstance(
            instance_id,
            name,
            instance.config_path,
            status_port,
            instance.allowed_modes,
        )
        index = self._instances.index(instance)
        self._instances[index] = updated
        self._save_instances()

        restarted = False
        if was_running and restart:
            mode = active.get("mode")
            if mode == "hatch-stage":
                stage = str(feature or "").removeprefix("hatch-stage-")
                if stage not in HATCH_STAGE_LABELS:
                    stage = "hatch"
                self.start("hatch-stage", stage=stage, instance_id=instance_id)
            elif mode in {"hunt", "hatch-beginner", "hatch-beginner-hunt", "hatch-hunt"}:
                self.start(str(mode), instance_id=instance_id)
            restarted = True
        return {
            "accepted": True,
            "action": "update-instance",
            "instance_id": instance_id,
            "name": name,
            "serial": serial,
            "status_port": status_port,
            "restarted": restarted,
            "message": "設定已儲存並重新啟動" if restarted else "設定已儲存",
        }

class _DashboardHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        metrics: MetricsStore,
        hatch_inventory: HatchBoostInventoryStore,
        controller: DashboardController,
        assets: Path,
        dashboard: DashboardServer,
    ) -> None:
        self.metrics = metrics
        self.hatch_inventory = hatch_inventory
        self.controller = controller
        self.assets = assets
        self.dashboard = dashboard
        super().__init__(address, _DashboardHandler)


class _DashboardHandler(BaseHTTPRequestHandler):
    server: _DashboardHttpServer

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_asset(self, name: str) -> None:
        path = (self.server.assets / name).resolve()
        if path.parent != self.server.assets.resolve() or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _ASSET_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length <= 0 or length > 1024:
            raise ValueError("JSON body must be between 1 and 1024 bytes")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self._send_asset("index.html")
        elif path == "/dashboard.css":
            self._send_asset("dashboard.css")
        elif path == "/dashboard.js":
            self._send_asset("dashboard.js")
        elif path == "/api/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "dino-dashboard",
                    "api_version": DASHBOARD_VERSION,
                    "process_id": os.getpid(),
                },
            )
        elif path == "/api/overview":
            selected = parse_qs(parsed.query).get("instance", [None])[0]
            self._send_json(200, self.server.dashboard._overview(selected))
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        origin = self.headers.get("Origin", "")
        if not _loopback_origin_allowed(origin):
            self._send_json(403, {"error": "forbidden_origin"})
            return
        if self.headers.get("X-Dino-Dashboard") != "1":
            self._send_json(403, {"error": "dashboard_header_required"})
            return
        parsed = urlparse(self.path)
        action = parsed.path.removeprefix("/api/control/")
        instance_id = parse_qs(parsed.query).get("instance", [None])[0]
        try:
            if action == "start-hunt":
                result = self.server.controller.start("hunt", instance_id=instance_id)
            elif action == "start-hatch-beginner":
                result = self.server.controller.start(
                    "hatch-beginner", instance_id=instance_id
                )
            elif action == "start-hatch-beginner-hunt":
                result = self.server.controller.start(
                    "hatch-beginner-hunt", instance_id=instance_id
                )
            elif action == "start-hatch-hunt":
                result = self.server.controller.start("hatch-hunt", instance_id=instance_id)
            elif action.startswith("start-stage-"):
                stage = action.removeprefix("start-stage-")
                result = self.server.controller.start(
                    "hatch-stage", stage=stage, instance_id=instance_id
                )
            elif action == "stop":
                result = self.server.controller.stop(instance_id)
            elif action == "scan-adb":
                result = self.server.controller.scan_adb(instance_id)
            elif action == "restart-game":
                result = self.server.controller.restart_game(instance_id)
            elif action == "restart-bot":
                result = self.server.controller.restart_bot(instance_id)
            elif action == "shutdown-dashboard":
                result = {
                    "accepted": True,
                    "action": action,
                    "message": "Dashboard shutdown requested",
                }

                def shutdown_later() -> None:
                    time.sleep(0.1)
                    self.server.shutdown()

                threading.Thread(target=shutdown_later, daemon=True).start()
            elif action == "set-boost-stock":
                payload = self._read_json()
                remaining = payload.get("remaining")
                if isinstance(remaining, bool) or not isinstance(remaining, int):
                    raise ValueError("remaining must be an integer")
                target = self.server.controller._instance(instance_id).instance_id
                inventory = self.server.dashboard._inventory_by_instance[target].set_remaining(
                    remaining
                )
                result = {
                    "accepted": True,
                    "action": action,
                    "inventory": inventory.as_dict(),
                    "message": f"冷卻加速券庫存已更新為 {inventory.remaining}",
                }
            elif action == "set-boost-enabled":
                payload = self._read_json()
                enabled = payload.get("enabled")
                if not isinstance(enabled, bool):
                    raise ValueError("enabled must be a boolean")
                target = self.server.controller._instance(instance_id).instance_id
                inventory = self.server.dashboard._inventory_by_instance[target].set_enabled(
                    enabled
                )
                result = {
                    "accepted": True,
                    "action": action,
                    "inventory": inventory.as_dict(),
                    "message": (
                        "下一輪孵化將使用冷卻加速券"
                        if enabled
                        else "已關閉冷卻加速券使用"
                    ),
                }
            elif action in {"diagnostics", "snapshot", "open-logs"}:
                result = self.server.controller.run_tool(action, instance_id)
            elif action == "add-instance":
                payload = self._read_json()
                result = self.server.controller.add_instance(
                    name=payload.get("name", ""),
                    serial=payload.get("serial", ""),
                    status_port=payload.get("status_port", 0),
                    instance_id=payload.get("id"),
                )
                self.server.dashboard._refresh_instance_stores()
            elif action == "update-instance":
                payload = self._read_json()
                target = instance_id or payload.get("id")
                if not isinstance(target, str) or not target:
                    raise ValueError("instance id is required")
                result = self.server.controller.update_instance(
                    instance_id=target,
                    name=payload.get("name", ""),
                    serial=payload.get("serial", ""),
                    status_port=payload.get("status_port", 0),
                    restart=payload.get("restart", True),
                )
            else:
                self._send_json(404, {"error": "not_found"})
                return
        except (RuntimeError, ValueError) as exc:
            self._send_json(409, {"error": str(exc), "action": action})
            return
        self._send_json(202, result)

    def log_message(self, format: str, *args: object) -> None:
        return


class DashboardServer:
    """Run the dashboard service on loopback."""

    def __init__(
        self,
        runtime_root: Path,
        logs_dir: Path,
        database: Path,
        *,
        port: int = 8780,
        assets: Path | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        self.logs_dir = logs_dir
        self.database = database
        self.port = port
        self.assets = assets or Path(__file__).with_name("dashboard_assets")
        self.controller = DashboardController(runtime_root, logs_dir, config_path=config_path)
        self._metrics_by_instance: dict[str, MetricsStore] = {}
        self._inventory_by_instance: dict[str, HatchBoostInventoryStore] = {}
        self._refresh_instance_stores()
        # Backwards-compatible handles for integrations that used the primary
        # instance's stores directly.
        primary = self.controller.instances[0].instance_id
        self.metrics = self._metrics_by_instance[primary]
        self.hatch_inventory = self._inventory_by_instance[primary]
        self._server: _DashboardHttpServer | None = None
        self._thread: threading.Thread | None = None
        self._auto_resume_thread: threading.Thread | None = None
        self._auto_resume_stop = threading.Event()

    @property
    def url(self) -> str:
        port = self._server.server_address[1] if self._server else self.port
        return f"http://127.0.0.1:{port}"

    AUTO_RESUME_MODE = "hatch-hunt"
    AUTO_RESUME_IDLE_SECONDS = 10.0

    def _refresh_instance_stores(self) -> None:
        for instance in self.controller.instances:
            self._metrics_by_instance.setdefault(
                instance.instance_id,
                MetricsStore(instance.database, instance.logs_dir),
            )
            self._inventory_by_instance.setdefault(
                instance.instance_id,
                HatchBoostInventoryStore(instance.database),
            )

    def _overview(self, selected_id: str | None = None) -> dict[str, Any]:
        self._refresh_instance_stores()
        definitions = self.controller.instances
        if len(definitions) == 1:
            # Keep the old monkeypatch/integration seam for single-instance
            # callers while the normal multi-instance path is explicit.
            active_items = [self.controller.discover()]
        else:
            active_items = self.controller.discover_all()
        by_id: dict[str, dict[str, Any]] = {}
        for item in active_items:
            # Older integrations may provide the pre-multi-instance discovery
            # shape; treat that result as the primary instance.
            item.setdefault("instance_id", definitions[0].instance_id)
            by_id[item["instance_id"]] = item
        instances: list[dict[str, Any]] = []
        for definition in definitions:
            active = by_id.get(definition.instance_id) or self.controller.discover(
                definition.instance_id
            )
            metrics = self._metrics_by_instance[definition.instance_id].snapshot()
            inventory = self._inventory_by_instance[definition.instance_id].snapshot()
            instances.append(
                {
                    "id": definition.instance_id,
                    "name": definition.name,
                    "serial": active.get("serial"),
                    "status_port": definition.status_port,
                    "allowed_modes": sorted(definition.allowed_modes),
                    "active": active,
                    "operation": self.controller.operation(definition.instance_id),
                    "metrics": metrics,
                    "hatch_boost_inventory": inventory.as_dict(),
                }
            )
        selected = next(
            (item for item in instances if item["id"] == selected_id),
            None,
        )
        if selected is None:
            selected = next((item for item in instances if item["active"]["running"]), instances[0])
        return {
            "selected_instance": selected["id"],
            "instances": instances,
            "active": selected["active"],
            "operation": selected["operation"],
            "metrics": selected["metrics"],
            "hatch_boost_inventory": selected["hatch_boost_inventory"],
        }

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _DashboardHttpServer(
            ("127.0.0.1", self.port),
            self.metrics,
            self.hatch_inventory,
            self.controller,
            self.assets,
            self,
        )
        if self._auto_resume_thread is None:
            self._auto_resume_thread = threading.Thread(
                target=self._watch_stage_completion, daemon=True
            )
            self._auto_resume_thread.start()

    def _watch_stage_completion(self) -> None:
        """Queue the continuous mode after a standalone stage finishes on its own.

        A stage that exits by itself leaves nothing running; if the user takes
        no action within the grace window, resume hatch-hunt. An explicit
        dashboard stop suppresses the resume - stopping means stopping.
        """

        last_mode: dict[str, str] = {}
        idle_since: dict[str, float] = {}
        while not self._auto_resume_stop.wait(3):
            try:
                active_items = self.controller.discover_all()
            except Exception:
                continue
            seen: set[str] = set()
            for active in active_items:
                instance_id = str(active["instance_id"])
                seen.add(instance_id)
                if active["running"]:
                    last_mode[instance_id] = str(active["mode"])
                    idle_since.pop(instance_id, None)
                    continue
                if last_mode.get(instance_id) != "hatch-stage":
                    idle_since.pop(instance_id, None)
                    continue
                now = time.monotonic()
                if now - self.controller._explicit_stop_at.get(instance_id, 0) < 60:
                    last_mode.pop(instance_id, None)
                    idle_since.pop(instance_id, None)
                    continue
                if now - self.controller._last_start.get(instance_id, 0) < 30:
                    idle_since.pop(instance_id, None)
                    continue
                if instance_id not in idle_since:
                    idle_since[instance_id] = now
                    continue
                if now - idle_since[instance_id] < self.AUTO_RESUME_IDLE_SECONDS:
                    continue
                last_mode.pop(instance_id, None)
                idle_since.pop(instance_id, None)
                with suppress(RuntimeError):
                    self.controller.start(self.AUTO_RESUME_MODE, instance_id=instance_id)
            for instance_id in (set(last_mode) | set(idle_since)) - seen:
                last_mode.pop(instance_id, None)
                idle_since.pop(instance_id, None)

    def serve_forever(self, *, open_browser: bool = False) -> None:
        self.start()
        assert self._server is not None
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(self.url)).start()
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()
            self._server = None

    def start_background(self) -> None:
        self.start()
        assert self._server is not None
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="dino-dashboard",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._auto_resume_stop.set()
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def __enter__(self) -> DashboardServer:
        self.start_background()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
