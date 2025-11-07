from __future__ import annotations

import logging
from typing import Any

from varipeps import config as _cfg_mod  # uses the global config instance

# --- Custom tqdm-based handlers ---

class TqdmUpdateHandler(logging.Handler):
    """Updates a tqdm progress bar's postfix string instead of printing."""
    def __init__(self, pbar: Any):
        super().__init__()
        self.pbar = pbar

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Truncate to keep the bar compact
            self.pbar.set_postfix_str(str(msg), refresh=True)
        except Exception:  # nosec - logging must never raise
            self.handleError(record)


class TqdmWriteHandler(logging.Handler):
    """Writes messages via tqdm.write (thread-safe)."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            from tqdm import tqdm
            tqdm.write(str(msg))
        except Exception:
            self.handleError(record)


class ExcludeLoggerFilter(logging.Filter):
    """Exclude records whose logger name starts with a given prefix."""
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(self.prefix)

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

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    use_tqdm = bool(getattr(cfg, "log_tqdm", False))

    if use_tqdm:
        fmt = logging.Formatter(fmt="%(message)s")
        # Console via tqdm.write for all varipeps loggers except optimizer
        tw = TqdmWriteHandler()
        tw.setFormatter(fmt)
        tw.addFilter(ExcludeLoggerFilter("varipeps.optimizer"))
        root.addHandler(tw)

        # Preserve file logging if enabled
        if getattr(cfg, "log_to_file", False):
            fh = logging.FileHandler(getattr(cfg, "log_file", "varipeps.log"))
            fh.setFormatter(fmt)
            root.addHandler(fh)

        # Optimizer uses a tqdm progress bar update handler
        opt_logger = logging.getLogger("varipeps.optimizer")
        for h in list(opt_logger.handlers):
            opt_logger.removeHandler(h)

        from tqdm import tqdm
        # Create a lightweight bar that we only update the postfix for
        pbar = tqdm(total=0, position=0, leave=True, dynamic_ncols=True)

        if pbar is not None:
            th = TqdmUpdateHandler(pbar)
            th.setFormatter(fmt)
            opt_logger.addHandler(th)

        # Keep propagation so optimizer still logs to file handler if present,
        # while console is suppressed by the ExcludeLoggerFilter on root.
        opt_logger.propagate = True
    else:
        # Standard console/file logging
        if getattr(cfg, "log_to_console", True):
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            root.addHandler(sh)

        if getattr(cfg, "log_to_file", False):
            fh = logging.FileHandler(getattr(cfg, "log_file", "varipeps.log"))
            fh.setFormatter(fmt)
            root.addHandler(fh)

        # Ensure optimizer has no leftover tqdm handler from a previous init
        opt_logger = logging.getLogger("varipeps.optimizer")
        for h in list(opt_logger.handlers):
            opt_logger.removeHandler(h)
        opt_logger.propagate = True

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
    logging.getLogger("varipeps.expectation").setLevel(
        _to_py_log_level(getattr(cfg, "log_level_expectation", logging.INFO))
    )

    _LOGGING_INITIALIZED = True

def ensure_logging_configured(cfg: Any | None = None) -> None:
    """
    Initialize logging once on first call; subsequent calls are no-ops.
    """
    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        init_logging(cfg)
