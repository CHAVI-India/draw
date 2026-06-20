"""JobQueue contract tests — run against BOTH backends (SQL + in-memory).

Both implementations must satisfy the same ``draw_contracts.JobQueue`` Protocol and
pass the same behavioural tests, which is the whole point of the abstraction: the
pipeline works unchanged regardless of backend.
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


def test_satisfies_protocol(queue):
    assert isinstance(queue, JobQueue)


def test_enqueue_dedup(queue):
    assert queue.enqueue("s1", "/in/s1", "TSPrime") is True
    assert queue.enqueue("s1", "/in/s1", "TSPrime") is False  # dup rejected
    assert queue.exists("s1")


def test_claim_marks_started_and_returns_items(queue):
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.enqueue("s2", "/in/s2", "TSPrime")
    claimed = queue.claim("TSPrime", limit=10)
    assert {c.series_name for c in claimed} == {"s1", "s2"}
    assert all(c.status == JobStatus.STARTED for c in claimed)
    # Re-claim yields nothing: items already moved out of INIT.
    assert queue.claim("TSPrime", limit=10) == []


def test_claim_is_model_scoped(queue):
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.enqueue("g1", "/in/g1", "TSGyne")
    assert [c.series_name for c in queue.claim("TSGyne", limit=10)] == ["g1"]


def test_transitions(queue):
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.claim("TSPrime", limit=1)
    queue.mark_predicted("s1", "/out/s1")
    pred = queue.list_by_status("TSPrime", JobStatus.PREDICTED, limit=10)
    assert pred and pred[0].output_path == "/out/s1"
    queue.mark_sent("s1")
    assert queue.list_by_status("TSPrime", JobStatus.SENT, limit=10)


def test_mark_failed(queue):
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.claim("TSPrime", limit=1)
    queue.mark_failed("s1", "boom")
    assert queue.list_by_status("TSPrime", JobStatus.FAILED, limit=10)


def test_no_double_claim_across_two_consumers_sql():
    """Concurrency: two consumers claiming the same SQL queue never get the same item.

    Each call to ``claim`` runs its own transaction; with batch_size=1 the two
    consumers must end up with disjoint items and cover all of them.
    """
    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    producer = SqlJobQueue(engine, batch_size=1)
    for i in range(6):
        producer.enqueue(f"s{i}", f"/in/s{i}", "TSPrime")

    worker_a = SqlJobQueue(engine, batch_size=1)
    worker_b = SqlJobQueue(engine, batch_size=1)

    claimed: list[str] = []
    turn = 0
    while True:
        w = worker_a if turn % 2 == 0 else worker_b
        items = w.claim("TSPrime", limit=1)
        if not items:
            # give the other worker a chance before declaring the queue drained
            other = worker_b if w is worker_a else worker_a
            items = other.claim("TSPrime", limit=1)
            if not items:
                break
        claimed.extend(i.series_name for i in items)
        turn += 1

    assert sorted(claimed) == [f"s{i}" for i in range(6)]
    assert len(claimed) == len(set(claimed))  # no item claimed twice
