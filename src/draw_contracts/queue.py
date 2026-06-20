"""Message-queue abstraction for segmentation work.

The pipeline treats persistence as a *queue*, not a database: producers ``enqueue``
studies, a worker ``claim``s a batch atomically, then marks each item
``predicted`` / ``sent`` / ``failed``. This Protocol is the only thing the watcher
and prediction loop depend on, so the backing store (SQLite, Postgres, an in-memory
dict, or a real broker like Redis/SQS later) is swappable without touching the
pipeline. Queue verbs replace the old DB-shaped method names.

``QueueItem`` is a plain DTO so the queue never leaks an ORM row to callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from draw_contracts.protocols import JobStatus


@dataclass(frozen=True)
class QueueItem:
    """One unit of queued work (a study to segment with a given model)."""

    series_name: str
    input_path: str
    model: str
    status: JobStatus = JobStatus.INIT
    output_path: str | None = None
    item_id: int | None = None
    attempts: int = 0


@runtime_checkable
class JobQueue(Protocol):
    """A durable work queue for segmentation jobs."""

    def enqueue(self, series_name: str, input_path: str, model: str) -> bool:
        """Add a study. Returns False if ``series_name`` is already queued (dedup)."""

    def exists(self, series_name: str) -> bool:
        """Whether a study with this series id is already known to the queue."""

    def claim(self, model: str, limit: int = 1) -> list[QueueItem]:
        """Atomically take up to ``limit`` pending items for ``model`` and mark them
        in-progress. Concurrent workers never claim the same item."""

    def list_by_status(self, model: str, status: JobStatus, limit: int = 1) -> list[QueueItem]:
        """Peek at items in a given state (e.g. to ship completed results)."""

    def mark_predicted(self, series_name: str, output_path: str) -> None:
        """Mark a study's segmentation done and record where the output landed."""

    def mark_sent(self, series_name: str) -> None:
        """Mark a study's results delivered downstream."""

    def mark_failed(self, series_name: str, error: str) -> None:
        """Mark a study failed (e.g. inference error)."""

    def requeue_expired(self, lease_seconds: int, max_attempts: int) -> int:
        """Crash recovery: re-queue items stuck in-progress past ``lease_seconds``.

        If a worker is killed mid-prediction, its claimed items would otherwise sit
        in STARTED forever. This returns expired leases to INIT (so another worker
        retries them) until ``max_attempts`` is exceeded, after which they go to
        FAILED. Returns the number of items acted on. Idempotent; safe to call on a
        timer from any worker.
        """
