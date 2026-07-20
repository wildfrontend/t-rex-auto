"""RAM-only BlueStacks capture backends."""

from __future__ import annotations

import ctypes
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import cv2
import mss
import numpy as np

from .actions import AdbClient, AdbError
from .models import Frame


class CaptureError(RuntimeError):
    pass


class BlueStacksWindowFinder:
    def __init__(
        self,
        title_fragments: Sequence[str],
        process_names: Sequence[str] = ("HD-Player.exe",),
    ):
        self.title_fragments = tuple(item.casefold() for item in title_fragments)
        self.process_names = frozenset(item.casefold() for item in process_names)

    @staticmethod
    def _process_name(hwnd: int) -> str:
        try:
            import win32process
        except ImportError:
            return ""
        _, process_id = win32process.GetWindowThreadProcessId(hwnd)
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id)
        if not handle:
            return ""
        try:
            buffer = ctypes.create_unicode_buffer(32768)
            size = ctypes.c_ulong(len(buffer))
            if ctypes.windll.kernel32.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(size)
            ):
                return Path(buffer.value).name.casefold()
            return ""
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    def find(self) -> int:
        if sys.platform != "win32":
            raise CaptureError("BlueStacks window capture requires Windows Python")
        try:
            import win32gui
        except ImportError as exc:
            raise CaptureError("pywin32 is required for BlueStacks window discovery") from exc

        candidates: list[tuple[int, int, str]] = []

        def visit(hwnd: int, _: object) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd).strip()
            title_matches = bool(title) and any(
                fragment in title.casefold() for fragment in self.title_fragments
            )
            process_matches = self._process_name(hwnd) in self.process_names
            if not title_matches and not process_matches:
                return
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            area = max(0, right - left) * max(0, bottom - top)
            if area:
                candidates.append((area, hwnd, title))

        win32gui.EnumWindows(visit, None)
        if not candidates:
            expected = ", ".join(self.title_fragments)
            raise CaptureError(f"No visible BlueStacks window found (expected title: {expected})")
        return max(candidates)[1]

    @staticmethod
    def activate(hwnd: int) -> None:
        try:
            import win32con
            import win32gui
        except ImportError as exc:
            raise CaptureError("pywin32 is required for BlueStacks window activation") from exc
        changed = win32gui.GetForegroundWindow() != hwnd or win32gui.IsIconic(hwnd)
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        if changed:
            with suppress(Exception):
                win32gui.BringWindowToTop(hwnd)
                win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.08)

    @staticmethod
    def client_region(hwnd: int) -> dict[str, int]:
        try:
            import win32gui
        except ImportError as exc:
            raise CaptureError("pywin32 is required for BlueStacks window capture") from exc
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            raise CaptureError("BlueStacks window has an empty client area")
        return {"left": screen_left, "top": screen_top, "width": width, "height": height}


class MssBlueStacksCapture:
    def __init__(
        self,
        window_titles: Sequence[str],
        process_names: Sequence[str] = ("HD-Player.exe",),
        viewport: tuple[int, int, int, int] | None = None,
        auto_viewport: bool = False,
        chrome_insets: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        if sys.platform == "win32":
            with suppress(AttributeError, OSError):
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
        self.finder = BlueStacksWindowFinder(window_titles, process_names)
        self.viewport = viewport
        self.auto_viewport = auto_viewport
        self.chrome_insets = chrome_insets
        self._sct = mss.mss()
        self._sequence = 0

    def capture(self) -> Frame:
        hwnd = self.finder.find()
        self.finder.activate(hwnd)
        region = self.finder.client_region(hwnd)
        if self.auto_viewport:
            left, top, right, bottom = self.chrome_insets
            width = region["width"] - left - right
            height = region["height"] - top - bottom
            if width <= 0 or height <= 0:
                raise CaptureError(
                    f"Chrome insets {self.chrome_insets} are outside client area "
                    f"{region['width']}x{region['height']}"
                )
            region = {
                "left": region["left"] + left,
                "top": region["top"] + top,
                "width": width,
                "height": height,
            }
        elif self.viewport is not None:
            x, y, width, height = self.viewport
            if x < 0 or y < 0 or x + width > region["width"] or y + height > region["height"]:
                raise CaptureError(
                    f"Configured viewport {self.viewport} is outside client area "
                    f"{region['width']}x{region['height']}"
                )
            region = {
                "left": region["left"] + x,
                "top": region["top"] + y,
                "width": width,
                "height": height,
            }
        raw = self._sct.grab(region)
        image = np.asarray(raw, dtype=np.uint8)[..., :3].copy()
        self._sequence += 1
        return Frame(image=image, source="bluestacks:mss", sequence=self._sequence)

    def close(self) -> None:
        self._sct.close()


class AdbScreencapCapture:
    def __init__(self, client: AdbClient):
        self.client = client
        self._sequence = 0

    def capture(self) -> Frame:
        try:
            payload = self.client.screencap_png()
        except AdbError as exc:
            raise CaptureError(str(exc)) from exc
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise CaptureError("ADB returned an invalid PNG screenshot")
        self._sequence += 1
        return Frame(image=image, source="bluestacks:adb", sequence=self._sequence)

    def close(self) -> None:
        pass
