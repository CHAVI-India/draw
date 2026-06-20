"""Both DB-setup paths work: auto-bootstrap (ensure_schema) and Alembic migrations."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import inspect  # noqa: E402

from draw_pipeline.dao.common import ensure_schema, get_engine  # noqa: E402


def test_ensure_schema_creates_table_and_is_idempotent(tmp_path):
    url = f"sqlite:///{tmp_path}/t.sqlite"
    engine = get_engine(url)
    ensure_schema(engine)
    ensure_schema(engine)  # second call must be a no-op, not an error

    cols = {c["name"] for c in inspect(engine).get_columns("dicomlog")}
    assert {"id", "series_name", "status", "model", "output_path"} <= cols


def test_alembic_upgrade_creates_table(tmp_path):
    pytest.importorskip("alembic")
    from draw_pipeline import migrations

    url = f"sqlite:///{tmp_path}/m.sqlite"
    migrations.upgrade(url, "head")
    assert inspect(get_engine(url)).has_table("dicomlog")


def test_status_enum_includes_failed():
    from draw_pipeline.dao.common import Status

    assert Status.FAILED.value == "FAILED"
