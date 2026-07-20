"""Human-readable daily log files without image or video side effects."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO


class DailyFileHandler(logging.Handler):
    def __init__(self, directory: Path, encoding: str = "utf-8") -> None:
        super().__init__()
        self.directory = directory
        self.encoding = encoding
        self._date = ""
        self._stream: TextIO | None = None

    def _ensure_stream(self) -> TextIO:
        today = datetime.now().strftime("%Y%m%d")
        if self._stream is None or self._date != today:
            if self._stream is not None:
                self._stream.close()
            self.directory.mkdir(parents=True, exist_ok=True)
            self._stream = (self.directory / f"{today}.log").open(
                "a", encoding=self.encoding, buffering=1
            )
            self._date = today
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = self._ensure_stream()
            stream.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


def configure_logging(logs_dir: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("dino_bot")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    file_handler = DailyFileHandler(logs_dir)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
