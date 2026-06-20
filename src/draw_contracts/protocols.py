"""Structural interfaces (PEP 544 Protocols) for the layer boundaries.

The core depends on these abstractions. Concrete implementations live in the
scaffolding layer (SQL sink + local FS storage today; Dynamo/S3 later) and are
injected at the entrypoint. No implementation needs to subclass these — a class
that matches the shape satisfies the Protocol, which keeps the core decoupled.
"""

from __future__ import annotations

import enum
from typing import Protocol, runtime_checkable

from draw_contracts.dto import SegmentationJob


class JobStatus(str, enum.Enum):
    """Lifecycle of a segmentation record. ``FAILED`` is new vs the legacy schema."""

    INIT = "INIT"
    STARTED = "STARTED"
    PREDICTED = "PREDICTED"
    SENT = "SENT"
    FAILED = "FAILED"


@runtime_checkable
class StatusSink(Protocol):
    """Where the core records per-series progress.

    Replaces the direct ``DBConnection.update_record_by_series_name`` call that
    used to live inside the NIfTI->RTStruct conversion. The core calls
    ``record_predicted`` once per produced series; the entrypoint decides whether
    that lands in SQL, DynamoDB, or nowhere (tests).
    """

    def record_predicted(self, series_name: str, output_path: str) -> None: ...

    def record_failed(self, job_id: str, error: str) -> None: ...


@runtime_checkable
class StorageBackend(Protocol):
    """Materializes job inputs locally and ships outputs out.

    The core only ever touches local filesystem paths. This boundary is what lets
    the same core run against a local directory (on-prem) or object storage (cloud)
    without changing a line of segmentation policy.
    """

    def fetch_input(self, job: SegmentationJob, dest_dir: str) -> str:
        """Make the job's input available under ``dest_dir``; return the local root."""

    def publish_output(self, local_dir: str, job: SegmentationJob) -> str:
        """Persist ``local_dir`` to the job's destination; return the final URI."""
