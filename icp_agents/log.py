"""Logging: what every agent is doing, every Gemini call and every page fetch.

Console shows INFO (progress you can follow); the run log file keeps DEBUG
(every call, retry and timing) for diagnosing a slow or stuck run.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT_CONSOLE = "%(asctime)s %(levelname)-5s %(name)-10s %(message)s"
LOG_FORMAT_FILE = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-12s %(message)s"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> Path | None:
    root = logging.getLogger("icp")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(LOG_FORMAT_CONSOLE, "%H:%M:%S"))
    root.addHandler(console)

    path = None
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FORMAT_FILE, "%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)
    # quiet noisy libraries, but keep their warnings
    for noisy in ("httpx", "httpcore", "google", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"icp.{name}")
