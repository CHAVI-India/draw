"""Database query/command layer for ``DicomLog`` records.

Ported from ``draw/dao/db.py``. Key changes:

* No global engine. ``DBConnection`` is constructed with an explicit engine (built
  from ``RuntimeEnv`` at the entrypoint), removing the import-time DB connection.
* ATOMIC DEQUEUE (bug fix). The legacy ``dequeue`` did ``top()`` (a SELECT) and then a
  *separate* per-row UPDATE to STARTED. Two pipeline workers could read the same INIT
  rows and both claim them (lost-update / double prediction). Here a single
  transaction selects-and-locks the candidate rows with
  ``with_for_update(skip_locked=True)`` and flips them to STARTED before commit, so a
  row is claimed by exactly one worker. ``skip_locked`` lets a second worker move past
  rows already locked by the first instead of blocking.

Method names (``dequeue``/``enqueue``/``top``/``exists``/``update_status_by_id``/
``update_record_by_series_name``) are kept for backward compatibility.
"""

from __future__ import annotations

from sqlalchemy import Engine, exists, select, update
from sqlalchemy.orm import Session

from draw_core.logging import get_logger
from draw_pipeline.dao.common import Status
from draw_pipeline.dao.table import DicomLog

log = get_logger(__name__)


class DBConnection:
    """Query the database and interact with ``DicomLog`` records."""

    def __init__(self, engine: Engine, batch_size: int = 1):
        self.engine = engine
        self.batch_size = batch_size

    def dequeue(self, model: str) -> list[DicomLog]:
        """Atomically claim a batch of INIT records for ``model`` and mark STARTED.

        Returns detached ``DicomLog`` snapshots for the claimed rows. Two workers
        running concurrently never claim the same row.
        """
        try:
            with Session(self.engine) as sess:
                stmt = (
                    select(DicomLog)
                    .where(DicomLog.model == model)
                    .where(DicomLog.status == Status.INIT)
                    .order_by(DicomLog.created_on)
                    .limit(self.batch_size)
                    .with_for_update(skip_locked=True)
                )
                rows = sess.scalars(stmt).all()
                for row in rows:
                    row.status = Status.STARTED
                sess.flush()
                claimed = [_detach(row) for row in rows]
                sess.commit()
            log.info("Dequeuing %d", len(claimed))
            return claimed
        except Exception:
            log.error("ERROR while dequeuing", exc_info=True)
            return []

    def exists(self, series_name: str) -> bool:
        with Session(self.engine) as sess:
            q = sess.query(exists().where(DicomLog.series_name == series_name))
            return bool(sess.execute(q).scalar())

    def top(self, model: str, status: Status) -> list[DicomLog]:
        try:
            with Session(self.engine) as sess:
                stmt = (
                    select(DicomLog)
                    .where(DicomLog.model == model)
                    .where(DicomLog.status == status)
                    .order_by(DicomLog.created_on)
                    .limit(self.batch_size)
                )
                return [_detach(r) for r in sess.scalars(stmt).all()]
        except Exception:
            log.error("Error while fetching TOP %s %s", status, model, exc_info=True)
            return []

    def enqueue(self, records: list[DicomLog]) -> None:
        try:
            n = len(records)
            with Session(self.engine) as sess:
                sess.add_all(records)
                sess.commit()
            log.info("Enqueued %d", n)
        except Exception:
            log.error("Could not insert records", exc_info=True)

    def update_status_by_id(self, dcm_log: DicomLog, updated_status: Status) -> None:
        with Session(self.engine) as sess:
            stmt = update(DicomLog).where(DicomLog.id == dcm_log.id).values(status=updated_status)
            sess.execute(stmt)
            sess.commit()

    def update_record_by_series_name(
        self, series_name: str, output_path: str, status: Status
    ) -> None:
        with Session(self.engine) as sess:
            stmt = (
                update(DicomLog)
                .where(DicomLog.series_name == series_name)
                .values(status=status, output_path=output_path)
            )
            sess.execute(stmt)
            sess.commit()


def _detach(row: DicomLog) -> DicomLog:
    """Return a session-independent copy of a row so callers can use it after commit."""
    return DicomLog(
        id=row.id,
        series_name=row.series_name,
        input_path=row.input_path,
        output_path=row.output_path,
        status=row.status,
        model=row.model,
        created_on=row.created_on,
    )
