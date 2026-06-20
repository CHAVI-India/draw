"""Adapt a ``JobQueue`` to the core's ``StatusSink`` Protocol.

The core records per-series progress through a ``StatusSink``; the pipeline persists
that progress in its ``JobQueue``. This thin adapter bridges the two so any queue
backend (SQL, in-memory, …) receives the core's status updates without the core
knowing anything about queues.
"""

from __future__ import annotations

from draw_contracts.queue import JobQueue
from draw_core.logging import get_logger

log = get_logger(__name__)


class QueueStatusSink:
    """Routes core status callbacks into a JobQueue. Satisfies ``StatusSink``."""

    def __init__(self, queue: JobQueue):
        self.queue = queue

    def record_predicted(self, series_name: str, output_path: str) -> None:
        self.queue.mark_predicted(series_name, output_path)

    def record_failed(self, job_id: str, error: str) -> None:
        # job_id carries the series_name in the pipeline path.
        self.queue.mark_failed(job_id, error)
