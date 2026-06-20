"""Plain data-transfer objects passed across layer boundaries.

These replace passing the SQLAlchemy ORM ``DicomLog`` into the core. The core
should never see persistence types — only these frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """Which segmentation model/protocol to run.

    ``name`` is a registry key (e.g. ``"TSPrime"``). ``only_original`` mirrors the
    existing flag: when True, RT-Struct files in the input are not parsed.
    """

    name: str
    only_original: bool = True


@dataclass(frozen=True)
class SegmentationJob:
    """One unit of work: segment a study with a model and write results out.

    ``input_uri`` / ``output_uri`` are opaque to the core — a storage backend
    resolves them to local paths at the edges (local dir today, ``s3://`` later).
    """

    job_id: str
    input_uri: str
    output_uri: str
    model: ModelSpec


@dataclass(frozen=True)
class SeriesResult:
    """One produced RT-Struct, keyed by the DICOM SeriesInstanceUID."""

    series_name: str
    output_path: str


@dataclass(frozen=True)
class SegmentationResult:
    """Everything one job produced, before any upload step."""

    job_id: str
    output_local_dir: str
    series: list[SeriesResult] = field(default_factory=list)
