"""In-memory JobQueue — the simplest possible backend.

No database at all. Useful for tests and for the simplest single-process
deployment where durability across restarts is not required. Thread-safe for the
single-watcher + single-worker model; not shared across processes.

Demonstrates the value of the JobQueue Protocol: the pipeline runs against this
unchanged, just by injecting it instead of SqlJobQueue.
"""

from __future__ import annotations

import threading

from draw_contracts.protocols import JobStatus
from draw_contracts.queue import QueueItem
from draw_core.logging import get_logger

log = get_logger(__name__)


class InMemoryJobQueue:
    """A dict-backed JobQueue. Satisfies the JobQueue Protocol."""

    def __init__(self) -> None:
        self._items: dict[str, QueueItem] = {}
        self._lock = threading.Lock()
        self._seq = 0

    def enqueue(self, series_name: str, input_path: str, model: str) -> bool:
        with self._lock:
            if series_name in self._items:
                return False
            self._seq += 1
            self._items[series_name] = QueueItem(
                series_name=series_name,
                input_path=input_path,
                model=model,
                status=JobStatus.INIT,
                item_id=self._seq,
            )
            return True

    def exists(self, series_name: str) -> bool:
        with self._lock:
            return series_name in self._items

    def claim(self, model: str, limit: int = 1) -> list[QueueItem]:
        with self._lock:
            pending = sorted(
                (
                    i
                    for i in self._items.values()
                    if i.model == model and i.status == JobStatus.INIT
                ),
                key=lambda i: i.item_id or 0,
            )[:limit]
            claimed = []
            for item in pending:
                started = QueueItem(**{**item.__dict__, "status": JobStatus.STARTED})
                self._items[item.series_name] = started
                claimed.append(started)
            return claimed

    def list_by_status(self, model: str, status: JobStatus, limit: int = 1) -> list[QueueItem]:
        with self._lock:
            return [
                i
                for i in self._items.values()
                if i.model == model and i.status == status
            ][:limit]

    def mark_predicted(self, series_name: str, output_path: str) -> None:
        self._update(series_name, JobStatus.PREDICTED, output_path=output_path)

    def mark_sent(self, series_name: str) -> None:
        self._update(series_name, JobStatus.SENT)

    def mark_failed(self, series_name: str, error: str) -> None:
        log.error("Job failed for %s: %s", series_name, error)
        self._update(series_name, JobStatus.FAILED)

    def _update(self, series_name: str, status: JobStatus, output_path: str | None = None) -> None:
        with self._lock:
            item = self._items.get(series_name)
            if item is None:
                return
            patch = {"status": status}
            if output_path is not None:
                patch["output_path"] = output_path
            self._items[series_name] = QueueItem(**{**item.__dict__, **patch})
