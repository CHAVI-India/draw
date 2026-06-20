"""Programmatic Alembic access so migrations are a CLI subcommand, not a chore.

Lets operators run ``draw db upgrade`` instead of remembering the bare ``alembic``
invocation with the right ``-c`` path and ``DRAW_DB_URL``. The Alembic config is
resolved from the packaged ``alembic.ini`` regardless of CWD.
"""

from __future__ import annotations

import os
from importlib import resources

from alembic import command
from alembic.config import Config

from draw_core.logging import get_logger

log = get_logger(__name__)


def _alembic_config(db_url: str) -> Config:
    # The packaged alembic.ini lives next to this module's package root.
    ini_path = resources.files("draw_pipeline").joinpath("alembic.ini")
    cfg = Config(str(ini_path))
    script_location = resources.files("draw_pipeline").joinpath("alembic")
    cfg.set_main_option("script_location", str(script_location))
    cfg.set_main_option("sqlalchemy.url", db_url)
    # env.py also reads this; set it so both paths agree.
    os.environ["DRAW_DB_URL"] = db_url
    return cfg


def upgrade(db_url: str, revision: str = "head") -> None:
    log.info("Alembic upgrade -> %s on %s", revision, db_url)
    command.upgrade(_alembic_config(db_url), revision)


def downgrade(db_url: str, revision: str) -> None:
    log.info("Alembic downgrade -> %s on %s", revision, db_url)
    command.downgrade(_alembic_config(db_url), revision)


def current(db_url: str) -> None:
    command.current(_alembic_config(db_url), verbose=True)
