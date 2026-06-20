"""Stable contracts shared across DRAW layers.

This package holds ONLY pure types and Protocols — no heavy dependencies, no I/O,
no import-time side effects. Both the high-level core (segmentation policy) and the
low-level scaffolding (DB, filesystem, S3) depend on these abstractions rather than
on each other. This is the Dependency Inversion boundary: details depend on policy.
"""

from draw_contracts.dto import ModelSpec, SegmentationJob, SegmentationResult, SeriesResult
from draw_contracts.protocols import JobStatus, StatusSink, StorageBackend
from draw_contracts.sink import NullStatusSink

__all__ = [
    "ModelSpec",
    "SegmentationJob",
    "SegmentationResult",
    "SeriesResult",
    "JobStatus",
    "StatusSink",
    "StorageBackend",
    "NullStatusSink",
]
