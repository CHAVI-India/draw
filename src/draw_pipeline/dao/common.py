"""SQLAlchemy base + engine factory.

Ported from ``draw/dao/common.py``. Changes:

* No engine is created at import time. The legacy module built ``DB_ENGINE`` from the
  env-derived ``DB_CONFIG`` on import, so importing the DAO crashed without an env
  file. Here ``get_engine(url)`` builds it on demand from an explicit URL.
* The hardcoded ``READ UNCOMMITTED`` isolation level is dropped (use the driver
  default); it was a foot-gun and not portable across backends (e.g. sqlite).
* The ``Status`` enum mirrors ``draw_contracts.JobStatus`` and now includes ``FAILED``.
* ``model`` is stored as a plain string column rather than a dynamically-built Enum,
  so the ORM no longer needs the model registry at import.
"""

from __future__ import annotations

import enum

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM tables."""


class Status(enum.Enum):
    """Mirror of ``draw_contracts.JobStatus`` for the persistence layer."""

    INIT = "INIT"
    STARTED = "STARTED"
    PREDICTED = "PREDICTED"
    SENT = "SENT"
    FAILED = "FAILED"


def get_engine(url: str, *, echo: bool = False) -> Engine:
    """Build a SQLAlchemy engine from an explicit URL (entrypoints only).

    Pooling args are only applied for backends that support them; sqlite (used by
    tests and the default on-prem config) ignores pool sizing.
    """
    kwargs: dict = {"echo": echo}
    if not url.startswith("sqlite"):
        kwargs.update(pool_size=10, max_overflow=20, pool_timeout=100)
    return create_engine(url, **kwargs)


def ensure_schema(engine: Engine) -> None:
    """Create any missing tables for the ORM metadata. Idempotent.

    This is the friendly auto-bootstrap path: the long-lived ``start-pipeline``
    process calls it at startup so a fresh deployment needs no manual
    ``alembic upgrade head`` to create the queue table. ``create_all`` only creates
    tables that don't already exist, so it is safe to call on every startup and is a
    no-op on an already-present table.

    Note: this creates the *current* schema; it does not perform incremental column
    migrations on an existing DB. Teams that evolve a long-lived prod schema should
    still use Alembic (``draw db upgrade``). For standing up a new single-machine
    instance, this is all that's needed.
    """
    # Imported here so DicomLog registers on Base.metadata without a circular import.
    from draw_pipeline.dao import table  # noqa: F401

    Base.metadata.create_all(engine)
