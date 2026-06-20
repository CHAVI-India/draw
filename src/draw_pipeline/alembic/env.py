"""Alembic migration environment for DRAW.

Rewritten to remove the legacy coupling. The old env.py imported ``draw.config``
(which read ``env.draw.yml`` at import) and a hardcoded ORM. This version:

* takes the DB URL from the ``DRAW_DB_URL`` env var if set, else falls back to the
  ``env.draw.yml`` via ``load_env`` — so ``draw db upgrade`` works without editing
  ``alembic.ini``;
* targets the live ORM metadata (``DicomLog``), so autogenerate stays correct as the
  schema evolves;
* does not configure application logging here (the CLI owns that).
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from draw_pipeline.dao import table  # noqa: F401  (registers DicomLog on Base.metadata)
from draw_pipeline.dao.common import Base

config = context.config


def _db_url() -> str:
    url = os.environ.get("DRAW_DB_URL")
    if url:
        return url
    # Fall back to the project env file so `draw db ...` needs no ini editing.
    from draw_pipeline.config import load_env

    return load_env().db_url


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _db_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
