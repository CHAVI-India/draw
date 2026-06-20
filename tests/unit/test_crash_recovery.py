"""Crash recovery: studies stranded in STARTED by a killed worker are recovered.

Models "the pipeline process was killed mid-prediction": an item is claimed (→ STARTED)
but never completed. A reaper must, once the lease expires, return it to the queue for
retry — and after too many attempts, fail it instead of retrying forever.

Time is injected (SQL: ``now=`` param; in-memory: logical ``advance``) so the tests are
deterministic and fast — no real sleeps.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

sa = pytest.importorskip("sqlalchemy")

from draw_contracts.protocols import JobStatus  # noqa: E402
from draw_pipeline.dao.common import Base  # noqa: E402
from draw_pipeline.dao.db import SqlJobQueue  # noqa: E402
from draw_pipeline.queue_memory import InMemoryJobQueue  # noqa: E402

LEASE = 600  # seconds


def _sql_queue() -> SqlJobQueue:
    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return SqlJobQueue(engine, batch_size=10)


# --- SQL backend (injected clock) ------------------------------------------------


def test_sql_unexpired_lease_is_not_requeued():
    queue = _sql_queue()
    queue.enqueue("s1", "/in/s1", "TSPrime")
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    queue.claim("TSPrime", limit=1)  # stamps claimed_at ~now (wall clock)

    # Reap "just after" claiming: lease not yet expired -> nothing happens.
    acted = queue.requeue_expired(LEASE, max_attempts=3, now=t0 + timedelta(seconds=LEASE - 1))

    # The item is still STARTED (not returned to INIT), so a fresh claim finds nothing.
    assert acted == 0
    assert queue.claim("TSPrime", limit=10) == []


def test_sql_expired_lease_is_requeued_for_retry():
    queue = _sql_queue()
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.claim("TSPrime", limit=1)

    acted = queue.requeue_expired(LEASE, max_attempts=3, now=datetime.now() + timedelta(days=1))

    assert acted == 1
    reclaimed = queue.claim("TSPrime", limit=10)
    assert [i.series_name for i in reclaimed] == ["s1"]


def test_sql_item_failed_after_max_attempts_exhausted():
    queue = _sql_queue()
    queue.enqueue("s1", "/in/s1", "TSPrime")
    future = datetime.now() + timedelta(days=1)

    # Two claims (attempts=2). With max_attempts=2, the next expiry should FAIL it.
    queue.claim("TSPrime", limit=1)
    queue.requeue_expired(LEASE, max_attempts=2, now=future)
    queue.claim("TSPrime", limit=1)

    acted = queue.requeue_expired(LEASE, max_attempts=2, now=future)

    assert acted == 1
    assert queue.list_by_status("TSPrime", JobStatus.FAILED, limit=10)
    assert queue.claim("TSPrime", limit=10) == []  # not retried again


# --- in-memory backend (logical clock) -------------------------------------------


def test_memory_expired_lease_is_requeued_for_retry():
    queue = InMemoryJobQueue()
    queue.enqueue("s1", "/in/s1", "TSPrime")
    queue.claim("TSPrime", limit=1)

    queue.advance(LEASE + 1)
    acted = queue.requeue_expired(LEASE, max_attempts=3)

    assert acted == 1
    assert [i.series_name for i in queue.claim("TSPrime", limit=10)] == ["s1"]


def test_memory_item_failed_after_max_attempts_exhausted():
    queue = InMemoryJobQueue()
    queue.enqueue("s1", "/in/s1", "TSPrime")

    queue.claim("TSPrime", limit=1)
    queue.advance(LEASE + 1)
    queue.requeue_expired(LEASE, max_attempts=2)  # attempt 1 -> requeue
    queue.claim("TSPrime", limit=1)
    queue.advance(LEASE + 1)
    acted = queue.requeue_expired(LEASE, max_attempts=2)  # attempt 2 hit cap -> fail

    assert acted == 1
    assert queue.list_by_status("TSPrime", JobStatus.FAILED, limit=10)
