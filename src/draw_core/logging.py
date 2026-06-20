"""Logging helpers with no import-time side effects.

Grounded in standard application/library logging guidance (see DRAW design notes):
library code should only *obtain* a logger and emit records; the *application*
entrypoint decides where logs go. The legacy ``draw.utils.logging`` violated this
by calling ``dictConfig`` at import and hardcoding ``logs/logfile.log`` relative to
the CWD — which crashed/created dirs merely on import and broke containers.

Usage:
    # in any library module, at module top — cheap, no side effects:
    from draw_core.logging import get_logger
    log = get_logger(__name__)

    # in an entrypoint, exactly once, before doing work:
    from draw_core.logging import configure_logging
    configure_logging()                      # stdout, INFO  (container default)
    configure_logging(logfile="logs/draw.log")  # also rotate to a file (on-prem)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys

_LOG_FORMAT = (
    "%(asctime)s PID:%(process)d [%(levelname)s] "
    "%(name)s.%(funcName)s:%(lineno)d: %(message)s"
)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module logger. No configuration, no side effects."""
    return logging.getLogger(name if name else "draw")


def configure_logging(
    level: str | int = "INFO",
    *,
    logfile: str | None = None,
    stream: bool = True,
) -> None:
    """Configure the root logger. Call once from an entrypoint (idempotent).

    Args:
        level: root log level (name or numeric).
        logfile: if set, also write to this path with daily rotation (7 backups).
            The parent directory is created if needed. Omit in containers — log to
            stdout and let the platform (CloudWatch/journald/ELK) handle the rest.
        stream: emit to stdout (default True).
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Idempotent: drop handlers we previously installed so repeated calls (e.g. in
    # tests or forked workers) don't duplicate output.
    for h in list(root.handlers):
        if getattr(h, "_draw_managed", False):
            root.removeHandler(h)

    formatter = logging.Formatter(_LOG_FORMAT)

    if stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        sh._draw_managed = True  # type: ignore[attr-defined]
        root.addHandler(sh)

    if logfile:
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        fh = logging.handlers.TimedRotatingFileHandler(
            logfile, when="midnight", interval=1, backupCount=7, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        fh._draw_managed = True  # type: ignore[attr-defined]
        root.addHandler(fh)
