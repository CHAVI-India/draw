"""SQL-backed JobQueue.

``SqlJobQueue`` implements the ``draw_contracts.JobQueue`` Protocol on top of the
``DicomLog`` table. The pipeline depends only on the Protocol, so this backend is
swappable for an in-memory queue (tests / single process) or a broker later, with no
pipeline changes.

The atomic ``claim`` is the key correctness property: a single transaction selects
candidate rows ``with_for_update(skip_locked=True)`` and flips them to STARTED before
commit, so two concurrent workers never claim the same study. (SKIP LOCKED is a no-op
on SQLite, which is single-writer anyway; it does real work on Postgres/MySQL.)

``QueueItem`` DTOs cross the boundary — the ORM ``DicomLog`` never leaks to callers.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import Engine, exists, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from draw_contracts.protocols import JobStatus
from draw_contracts.queue import QueueItem
from draw_core.logging import get_logger
from draw_pipeline.dao.common import Status
from draw_pipeline.dao.table import DicomLog

log = get_logger(__name__)


def _to_item(row: DicomLog) -> QueueItem:
    return QueueItem(
        series_name=row.series_name,
        input_path=row.input_path,
        model=row.model,
        status=JobStatus(row.status.value),
        output_path=row.output_path,
        item_id=row.id,
        attempts=row.attempts or 0,
    )


class SqlJobQueue:
    """SQL implementation of the JobQueue Protocol (SQLite/Postgres/MySQL via URL)."""

    def __init__(self, engine: Engine, batch_size: int = 1):
        self.engine = engine
        self.batch_size = batch_size

    # ---------------------------------------------------------------- producer
    def enqueue(self, series_name: str, input_path: str, model: str) -> bool:
        """Insert a study. Returns False if already present (dedup on series_name).

        Deduplication relies solely on the ``series_name`` UNIQUE constraint, not a
        prior ``exists()`` check: the check-then-insert pattern is a TOCTOU race (two
        watcher processes could both pass the check and then one fails). Catching the
        constraint violation makes the dedup atomic and treats a duplicate as the
        benign INFO it is — not a logged error.
        """
        try:
            with Session(self.engine) as sess:
                sess.add(
                    DicomLog(
                        series_name=series_name,
                        input_path=input_path,
                        model=model,
                        status=Status.INIT,
                    )
                )
                sess.commit()
            log.info("Enqueued %s (%s)", series_name, model)
            return True
        except IntegrityError:
            log.info("Skip enqueue; series already queued: %s", series_name)
            return False
        except Exception:
            log.error("Could not enqueue %s", series_name, exc_info=True)
            return False

    def exists(self, series_name: str) -> bool:
        with Session(self.engine) as sess:
            stmt = select(exists().where(DicomLog.series_name == series_name))
            return bool(sess.execute(stmt).scalar())

    # ---------------------------------------------------------------- consumer
    def claim(self, model: str, limit: int | None = None) -> list[QueueItem]:
        """Atomically claim up to ``limit`` INIT items for ``model``, mark STARTED."""
        limit = self.batch_size if limit is None else limit
        try:
            with Session(self.engine) as sess:
                stmt = (
                    select(DicomLog)
                    .where(DicomLog.model == model)
                    .where(DicomLog.status == Status.INIT)
                    .order_by(DicomLog.created_on)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
                rows = sess.scalars(stmt).all()
                claimed_at = datetime.now()
                for row in rows:
                    row.status = Status.STARTED
                    row.claimed_at = claimed_at
                    row.attempts = (row.attempts or 0) + 1
                sess.flush()
                items = [_to_item(row) for row in rows]
                sess.commit()
            log.info("Claimed %d for %s", len(items), model)
            return items
        except Exception:
            log.error("Error while claiming for %s", model, exc_info=True)
            return []

    def requeue_expired(
        self, lease_seconds: int, max_attempts: int, now: datetime | None = None
    ) -> int:
        """Re-queue (or fail) items whose lease expired because a worker died.

        ``now`` is injectable for deterministic tests; defaults to the wall clock.
        """
        now = now or datetime.now()
        cutoff = now - timedelta(seconds=lease_seconds)
        acted = 0
        try:
            with Session(self.engine) as sess:
                stmt = (
                    select(DicomLog)
                    .where(DicomLog.status == Status.STARTED)
                    .where(DicomLog.claimed_at.is_not(None))
                    .where(DicomLog.claimed_at < cutoff)
                    .with_for_update(skip_locked=True)
                )
                for row in sess.scalars(stmt).all():
                    if (row.attempts or 0) >= max_attempts:
                        row.status = Status.FAILED
                        log.error(
                            "Series %s exceeded %d attempts; marking FAILED",
                            row.series_name, max_attempts,
                        )
                    else:
                        row.status = Status.INIT
                        row.claimed_at = None
                        log.warning(
                            "Re-queuing stranded series %s (attempt %d)",
                            row.series_name, row.attempts,
                        )
                    acted += 1
                sess.commit()
            return acted
        except Exception:
            log.error("Error while requeuing expired leases", exc_info=True)
            return 0

    def list_by_status(
        self, model: str, status: JobStatus, limit: int | None = None
    ) -> list[QueueItem]:
        limit = self.batch_size if limit is None else limit
        try:
            with Session(self.engine) as sess:
                stmt = (
                    select(DicomLog)
                    .where(DicomLog.model == model)
                    .where(DicomLog.status == Status(status.value))
                    .order_by(DicomLog.created_on)
                    .limit(limit)
                )
                return [_to_item(r) for r in sess.scalars(stmt).all()]
        except Exception:
            log.error("Error listing %s for %s", status, model, exc_info=True)
            return []

    # ------------------------------------------------------------ transitions
    def mark_predicted(self, series_name: str, output_path: str) -> None:
        self._set_status(series_name, Status.PREDICTED, output_path=output_path)

    def mark_sent(self, series_name: str) -> None:
        self._set_status(series_name, Status.SENT)

    def mark_failed(self, series_name: str, error: str) -> None:
        log.error("Job failed for %s: %s", series_name, error)
        self._set_status(series_name, Status.FAILED)

    def _set_status(
        self, series_name: str, status: Status, output_path: str | None = None
    ) -> None:
        values: dict = {"status": status}
        if output_path is not None:
            values["output_path"] = output_path
        with Session(self.engine) as sess:
            sess.execute(
                update(DicomLog).where(DicomLog.series_name == series_name).values(**values)
            )
            sess.commit()
