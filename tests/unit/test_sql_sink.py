"""SqlStatusSink against in-memory sqlite using the real ORM."""

from __future__ import annotations

import pytest

sa = pytest.importorskip("sqlalchemy")

from draw_pipeline.dao.common import Base, Status  # noqa: E402
from draw_pipeline.dao.db import DBConnection  # noqa: E402
from draw_pipeline.dao.table import DicomLog  # noqa: E402
from draw_pipeline.sinks.sql_sink import SqlStatusSink  # noqa: E402


@pytest.fixture
def db():
    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return DBConnection(engine)


def test_record_predicted_updates_status_and_path(db):
    with sa.orm.Session(db.engine) as sess:
        sess.add(
            DicomLog(
                series_name="1.2.3",
                input_path="/in/study",
                model="TSPrime",
                status=Status.INIT,
            )
        )
        sess.commit()

    sink = SqlStatusSink(db)
    sink.record_predicted("1.2.3", "/out/study")

    with sa.orm.Session(db.engine) as sess:
        row = sess.query(DicomLog).filter_by(series_name="1.2.3").one()
        assert row.status == Status.PREDICTED
        assert row.output_path == "/out/study"


def test_record_failed_sets_failed(db):
    with sa.orm.Session(db.engine) as sess:
        sess.add(
            DicomLog(series_name="9.9", input_path="/in", model="TSPrime", status=Status.STARTED)
        )
        sess.commit()

    SqlStatusSink(db).record_failed("9.9", "boom")

    with sa.orm.Session(db.engine) as sess:
        row = sess.query(DicomLog).filter_by(series_name="9.9").one()
        assert row.status == Status.FAILED
