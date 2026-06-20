"""JobQueue behaviour — run against BOTH backends (SQL-on-sqlite + in-memory).

Both implementations satisfy ``draw_contracts.JobQueue`` and pass the same
behavioural tests, which is the point of the abstraction: the pipeline works
unchanged regardless of backend.

Scope note on concurrency: the real cross-worker no-double-claim guarantee relies on
``SELECT ... FOR UPDATE SKIP LOCKED``, which Postgres/MySQL honour but **SQLite
silently drops** (SQLite serialises writers anyway). So a SQLite-based test cannot
prove the locking semantics — asserting it here would be false coverage. That
guarantee is verified separately against a real Postgres in
``test_job_queue_postgres.py`` (skipped unless ``DRAW_TEST_POSTGRES_URL`` is set).
"""

from __future__ import annotations

import pytest

sa = pytest.importorskip("sqlalchemy")

from draw_contracts.protocols import JobStatus  # noqa: E402
from draw_contracts.queue import JobQueue  # noqa: E402
from draw_pipeline.dao.common import Base  # noqa: E402
from draw_pipeline.dao.db import SqlJobQueue  # noqa: E402
from draw_pipeline.queue_memory import InMemoryJobQueue  # noqa: E402


def _sql_queue() -> SqlJobQueue:
    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return SqlJobQueue(engine, batch_size=10)


@pytest.fixture(params=["sql", "memory"])
def queue(request) -> JobQueue:
    return _sql_queue() if request.param == "sql" else InMemoryJobQueue()


def test_implementation_satisfies_job_queue_protocol(queue):
    assert isinstance(queue, JobQueue)


def test_enqueuing_same_series_twice_is_rejected(queue):
    first = queue.enqueue("s1", "/in/s1", "TSPrime")
    second = queue.enqueue("s1", "/in/s1", "TSPrime")

    assert first is True
    assert second is False
    assert queue.exists("s1")


def test_claiming_returns_pending_items_marked_started(queue):
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.enqueue("s2", "/in/s2", "TSPrime")

    claimed = queue.claim("TSPrime", limit=10)

    assert {c.series_name for c in claimed} == {"s1", "s2"}
    assert all(c.status == JobStatus.STARTED for c in claimed)


def test_claimed_items_are_not_claimed_again(queue):
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.claim("TSPrime", limit=10)

    second_claim = queue.claim("TSPrime", limit=10)

    assert second_claim == []


def test_claiming_only_returns_items_for_the_requested_model(queue):
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.enqueue("g1", "/in/g1", "TSGyne")

    claimed = queue.claim("TSGyne", limit=10)

    assert [c.series_name for c in claimed] == ["g1"]


def test_marking_predicted_records_status_and_output_path(queue):
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.claim("TSPrime", limit=1)

    queue.mark_predicted("s1", "/out/s1")

    predicted = queue.list_by_status("TSPrime", JobStatus.PREDICTED, limit=10)
    assert len(predicted) == 1
    assert predicted[0].output_path == "/out/s1"


def test_marking_sent_moves_item_to_sent(queue):
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.claim("TSPrime", limit=1)
    queue.mark_predicted("s1", "/out/s1")

    queue.mark_sent("s1")

    assert queue.list_by_status("TSPrime", JobStatus.SENT, limit=10)


def test_marking_failed_moves_item_to_failed(queue):
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.claim("TSPrime", limit=1)

    queue.mark_failed("s1", "boom")

    assert queue.list_by_status("TSPrime", JobStatus.FAILED, limit=10)
