"""Unit tests for the typed db.json sidecar model."""

from __future__ import annotations

import json

from draw_core.constants import DB_NAME
from draw_core.sidecar import (
    SampleRecord,
    append_sidecar,
    find_sample,
    load_sidecar,
)


def test_roundtrip_preserves_legacy_json_keys(tmp_path):
    append_sidecar(str(tmp_path), SampleRecord(720, "000", "/data/ct/a"))
    append_sidecar(str(tmp_path), SampleRecord(720, "001", "/data/ct/b"))

    # On-disk keys must match the legacy format for backward compatibility.
    raw = json.loads((tmp_path / DB_NAME).read_text())
    assert raw[0] == {"DatasetID": 720, "SampleNumber": "000", "DICOMRootDir": "/data/ct/a"}

    records = load_sidecar(str(tmp_path))
    assert len(records) == 2
    assert find_sample(records, 720, "001").dicom_root_dir == "/data/ct/b"
    assert find_sample(records, 999, "000") is None


def test_load_missing_sidecar_returns_empty(tmp_path):
    assert load_sidecar(str(tmp_path)) == []
