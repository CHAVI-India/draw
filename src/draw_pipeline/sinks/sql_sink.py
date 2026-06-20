"""SQL-backed ``StatusSink``.

This is the concrete persistence side of the boundary the core only knows as a
``StatusSink`` Protocol. It moves the DB write that used to live *inside* the NIfTI
->RTStruct conversion out to the scaffolding layer.
"""

from __future__ import annotations

from draw_core.logging import get_logger
from draw_pipeline.dao.common import Status
from draw_pipeline.dao.db import DBConnection

log = get_logger(__name__)


class SqlStatusSink:
    """Records per-series status to the ``DicomLog`` table. Satisfies ``StatusSink``."""

    def __init__(self, db: DBConnection):
        self.db = db

    def record_predicted(self, series_name: str, output_path: str) -> None:
        self.db.update_record_by_series_name(series_name, output_path, Status.PREDICTED)

    def record_failed(self, job_id: str, error: str) -> None:
        # job_id carries the series_name here; no output produced on failure.
        log.error("Segmentation failed for %s: %s", job_id, error)
        self.db.update_record_by_series_name(job_id, "", Status.FAILED)
