"""Human-readable daily log files without image or video side effects."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO


class DailyFileHandler(logging.Handler):
    """Write one file per day, rolling over once at a size cap.

    A verbose run writes about 40 MB in seven hours, and the file was the only
    unbounded thing left in the app once the event stream learned to roll. Size
    is not only a disk question: everything that reads the log back - the
    control window's counters, the diagnostic bundle - pays for every line, and
    a status query that misses its two-second timeout shows raw log lines where
    the operator asked for numbers.
    """

    def __init__(
        self,
        directory: Path,
        encoding: str = "utf-8",
        max_bytes: int = 0,
    ) -> None:
        super().__init__()
        self.directory = directory
        self.encoding = encoding
        self.max_bytes = max(0, max_bytes)
        self._date = ""
        self._stream: TextIO | None = None
        self._written = 0

    def _ensure_stream(self) -> TextIO:
        today = datetime.now().strftime("%Y%m%d")
        if self._stream is None or self._date != today:
            if self._stream is not None:
                self._stream.close()
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{today}.log"
            self._written = path.stat().st_size if path.exists() else 0
            self._stream = path.open("a", encoding=self.encoding, buffering=1)
            self._date = today
        elif self.max_bytes and self._written >= self.max_bytes:
            # Roll rather than truncate, and keep exactly one generation: the
            # readers take the newest lines, and a run long enough to fill the
            # cap has already been diagnosed from the event stream if it needed
            # diagnosing at all.
            self._stream.close()
            path = self.directory / f"{today}.log"
            previous = self.directory / f"{today}.1.log"
            try:
                previous.unlink(missing_ok=True)
                path.replace(previous)
            except OSError:
                pass
            self._stream = path.open("w", encoding=self.encoding, buffering=1)
            self._written = 0
        assert self._stream is not None
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + "\n"
            stream = self._ensure_stream()
            stream.write(line)
            self._written += len(line.encode(self.encoding, errors="replace"))
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


def configure_logging(
    logs_dir: Path,
    verbose: bool = False,
    max_bytes: int = 32 * 1024 * 1024,
) -> logging.Logger:
    logger = logging.getLogger("dino_bot")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    file_handler = DailyFileHandler(logs_dir, max_bytes=max_bytes)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
