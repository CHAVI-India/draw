"""Typed model + I/O for the per-dataset ``db.json`` sidecar.

The sidecar maps an nnU-Net sample number back to the source DICOM directory so
the NIfTI->RTStruct step can find the reference series. It used to be passed
around as raw dicts indexed by string keys (``rec["DICOMRootDir"]``), which is
stringly-typed and error-prone. ``SampleRecord`` gives it a schema and one place
to change the on-disk format. The JSON field names are preserved for backward
compatibility with sidecars written by the legacy code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from draw_core.constants import DB_NAME


@dataclass(frozen=True)
class SampleRecord:
    """One row of the sidecar: which DICOM dir a sample number came from."""

    dataset_id: int
    sample_number: str
    dicom_root_dir: str

    # On-disk JSON keys (kept identical to the legacy format).
    _KEY_DATASET_ID = "DatasetID"
    _KEY_SAMPLE_NUMBER = "SampleNumber"
    _KEY_DICOM_ROOT_DIR = "DICOMRootDir"

    def to_dict(self) -> dict:
        return {
            self._KEY_DATASET_ID: self.dataset_id,
            self._KEY_SAMPLE_NUMBER: self.sample_number,
            self._KEY_DICOM_ROOT_DIR: self.dicom_root_dir,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> SampleRecord:
        return cls(
            dataset_id=int(raw[cls._KEY_DATASET_ID]),
            sample_number=str(raw[cls._KEY_SAMPLE_NUMBER]),
            dicom_root_dir=str(raw[cls._KEY_DICOM_ROOT_DIR]),
        )


def sidecar_path(dataset_dir: str) -> str:
    return os.path.normpath(f"{dataset_dir}/{DB_NAME}")


def load_sidecar(dataset_dir: str) -> list[SampleRecord]:
    path = sidecar_path(dataset_dir)
    if not os.path.exists(path):
        return []
    with open(path) as fp:
        return [SampleRecord.from_dict(r) for r in json.load(fp)]


def append_sidecar(dataset_dir: str, record: SampleRecord) -> None:
    records = load_sidecar(dataset_dir)
    records.append(record)
    with open(sidecar_path(dataset_dir), "w") as fp:
        json.dump([r.to_dict() for r in records], fp, indent=4)


def find_sample(
    records: list[SampleRecord], dataset_id: int, sample_number: str
) -> SampleRecord | None:
    for rec in records:
        if rec.dataset_id == int(dataset_id) and rec.sample_number == str(sample_number):
            return rec
    return None
