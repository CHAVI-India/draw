"""Local-filesystem ``StorageBackend``.

On-prem today the inputs and outputs are already local directories, so this backend
is a no-op pass-through. The boundary exists so the same core can later run against
object storage (S3/GCS) by swapping this implementation.
"""

from __future__ import annotations

from draw_contracts.dto import SegmentationJob


class LocalFsStorage:
    """Inputs/outputs are local paths already; nothing to fetch or upload."""

    def fetch_input(self, job: SegmentationJob, dest_dir: str) -> str:
        return job.input_uri

    def publish_output(self, local_dir: str, job: SegmentationJob) -> str:
        return local_dir
