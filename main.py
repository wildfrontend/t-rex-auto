"""Repository-local entry point; use `python main.py <command>`."""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from dino_bot.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
