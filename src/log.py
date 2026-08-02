"""
Project-wide logging.

Usage:
    from src.log import get_logger
    logger = get_logger(__name__)

Configured on first use: a stream handler plus a file handler writing to
logs/<timestamp>_<entry>[_rankN].log at the project root. Under torchrun,
non-zero ranks are set to WARNING so INFO lines (training progress,
checkpoints, ...) are emitted once instead of world_size times.
"""

import logging
import os
import sys
import time
from pathlib import Path

_ROOT = "mini_agnes"
_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _log_file():
    entry = Path(sys.argv[0]).stem or "console"
    rank = int(os.environ.get("RANK", "0"))
    suffix = f"_rank{rank}" if rank else ""
    return _LOG_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{entry}{suffix}.log"


def _configure():
    root = logging.getLogger(_ROOT)
    if root.handlers:
        return
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    _LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(_log_file())
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    is_rank0 = int(os.environ.get("RANK", "0")) == 0
    root.setLevel(logging.INFO if is_rank0 else logging.WARNING)
    root.propagate = False


def get_logger(name=None):
    _configure()
    if not name:
        return logging.getLogger(_ROOT)
    # Keep names short: src.training.trainer -> trainer
    short = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"{_ROOT}.{short}")
