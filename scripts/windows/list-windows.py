"""Development diagnostic: list large visible Windows windows and owners."""

from __future__ import annotations

import ctypes
from pathlib import Path

import win32gui
import win32process


def process_path(hwnd: int) -> str:
    _, process_id = win32process.GetWindowThreadProcessId(hwnd)
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id)
    if not handle:
        return "<access denied>"
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = ctypes.c_ulong(len(buffer))
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return buffer.value
        return "<unknown>"
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def visit(hwnd: int, _: object) -> None:
    if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
        return
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    if (right - left) * (bottom - top) < 100_000:
        return
    path = process_path(hwnd)
    print(
        hwnd,
        f"{right-left}x{bottom-top}",
        repr(win32gui.GetWindowText(hwnd)),
        repr(win32gui.GetClassName(hwnd)),
        Path(path).name,
    )


win32gui.EnumWindows(visit, None)
