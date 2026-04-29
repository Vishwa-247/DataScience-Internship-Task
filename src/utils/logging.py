"""Structured logging: rotating file handler + stdout, single configuration call.

Note: this module shadows the stdlib name `logging` only when imported as
`src.utils.logging`. Inside this file, `import logging` still resolves to the
stdlib via Python 3's absolute-import semantics.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "forecast",
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """Return a configured logger.

    The first call for a given `name` configures a rotating file handler at
    `<log_dir>/app.log` plus a stdout stream handler. Subsequent calls return
    the same logger without re-adding handlers (idempotent).

    Args:
        name: Logger name (use module path, e.g. ``"src.training.pipeline"``).
        log_dir: Directory for rotating log file. Created if missing.
        level: Logging level; default INFO.
        max_bytes: Per-file rotation threshold.
        backup_count: Number of rotated files to retain.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path / "app.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
