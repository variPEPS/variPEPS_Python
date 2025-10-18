from __future__ import annotations

import logging
from typing import Any

from varipeps import config as _cfg_mod  # uses the global config instance

_LOGGING_INITIALIZED = False

def _to_py_log_level(level: Any) -> int:
    # Accept both enum values and raw ints; OFF disables effectively
    try:
        val = int(level)
    except Exception:
        val = logging.INFO
    if val == 0:  # OFF
        return logging.CRITICAL + 10
    return val

def init_logging(cfg: Any | None = None) -> None:
    """
    Initialize logging based on the provided config (or global config).
    Safe to call multiple times; replaces handlers to avoid duplicates.
    """
    global _LOGGING_INITIALIZED
    if cfg is None:
        cfg = _cfg_mod.config

    root = logging.getLogger("varipeps")
    # Remove old handlers to prevent duplicate logs
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(_to_py_log_level(getattr(cfg, "log_level_global", logging.INFO)))
    root.propagate = False

    # fmt = logging.Formatter(
    #     fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
    #     datefmt="%H:%M:%S",
    # )

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if getattr(cfg, "log_to_console", True):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if getattr(cfg, "log_to_file", False):
        fh = logging.FileHandler(getattr(cfg, "log_file", "varipeps.log"))
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Per-module levels
    logging.getLogger("varipeps.optimizer").setLevel(
        _to_py_log_level(getattr(cfg, "log_level_optimizer", logging.INFO))
    )
    logging.getLogger("varipeps.ctmrg").setLevel(
        _to_py_log_level(getattr(cfg, "log_level_ctmrg", logging.INFO))
    )
    logging.getLogger("varipeps.line_search").setLevel(
        _to_py_log_level(getattr(cfg, "log_level_line_search", logging.INFO))
    )

    _LOGGING_INITIALIZED = True

def ensure_logging_configured(cfg: Any | None = None) -> None:
    """
    Initialize logging once on first call; subsequent calls are no-ops.
    """
    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        init_logging(cfg)
