"""Runtime environment config loaded from ``env.draw.yml`` (entrypoints only).

The legacy ``draw/config.py`` read this file at import time, which crashed any import
without the file present. Here it is parsed explicitly by entrypoints into a typed
``RuntimeEnv``. The YAML keys are unchanged for backward compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml
from schema import Schema, SchemaError

from draw_core.logging import get_logger

log = get_logger(__name__)

ENV_FILE_NAME = "env.draw.yml"

# Same keys/validation as the legacy draw/utils/mapping.py ENV_SCHEMA.
ENV_SCHEMA = Schema(
    {
        "DB_URL": str,
        "DB_NAME": str,
        "TABLE_NAME": str,
        "WATCH_DIR": str,
        "MODEL_DEF_ROOT": str,
    }
)


@dataclass(frozen=True)
class RuntimeEnv:
    """The on-disk environment config for the continuous pipeline."""

    db_url: str
    db_name: str
    table_name: str
    watch_dir: str
    model_def_root: str


def load_env(path: str = ENV_FILE_NAME) -> RuntimeEnv:
    """Read + schema-validate the env YAML into a ``RuntimeEnv``. Entrypoints only."""
    with open(path) as stream:
        raw = yaml.safe_load(stream)
    try:
        ENV_SCHEMA.validate(raw)
    except SchemaError:
        log.error("Invalid env config at %s", path, exc_info=True)
        raise
    return RuntimeEnv(
        db_url=raw["DB_URL"],
        db_name=raw["DB_NAME"],
        table_name=raw["TABLE_NAME"],
        watch_dir=os.path.normpath(raw["WATCH_DIR"]),
        model_def_root=os.path.normpath(raw["MODEL_DEF_ROOT"]),
    )
