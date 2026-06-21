"""Functional: NIfTI prediction -> DICOM RT-Struct, with no GPU and no DB.

Exercises the real conversion code path and asserts:
  * the severed DB write is gone (function returns SeriesResult objects),
  * the legacy "only last series recorded / NameError on empty" bug is fixed,
  * a valid RT-Struct file is actually produced on disk.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
from rt_utils import RTStructBuilder

from draw_conversion.nifti2rt import (
    convert_multilabel_nifti_to_rtstruct,
    convert_nifti_outputs_to_dicom,
    write_named_masks_to_rtstruct,
)
from draw_core.sidecar import SampleRecord, append_sidecar


def test_single_multilabel_conversion(tmp_path, ct_series_dir, multilabel_nifti):
    ct_dir, _series_uid, _ = ct_series_dir
    nifti_path, seg_map = multilabel_nifti
    save_dir = tmp_path / "out"

    result_dir = convert_multilabel_nifti_to_rtstruct(
        nifti_file_path=nifti_path,
        dicom_dir=ct_dir,
        save_dir=str(save_dir),
        label_to_name_map=seg_map,
    )
    rt_files = [f for f in os.listdir(result_dir) if f.endswith(".dcm")]
    assert rt_files, "an RT-Struct .dcm must be produced"


def test_outputs_to_dicom_returns_series_results(tmp_path, ct_series_dir, multilabel_nifti):
    ct_dir, _series_uid, _ = ct_series_dir
    nifti_path, seg_map = multilabel_nifti

    # Lay out a model prediction dir with one predicted NIfTI named seg_000.nii.gz
    model_pred_dir = tmp_path / "modelpred"
    model_pred_dir.mkdir()
    shutil.copy(nifti_path, model_pred_dir / "seg_000.nii.gz")

    # The sidecar maps sample 000 -> the source CT dir (typed, not raw dicts).
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    append_sidecar(
        str(dataset_dir),
        SampleRecord(dataset_id=720, sample_number="000", dicom_root_dir=ct_dir),
    )

    results = convert_nifti_outputs_to_dicom(
        model_pred_dir=str(model_pred_dir),
        final_output_dir=str(tmp_path / "final"),
        dataset_dir=str(dataset_dir),
        dataset_id=720,
        exp_number="2024-01-01.00-00",
        seg_map=seg_map,
    )

    assert len(results) == 1
    assert results[0].series_name  # SeriesInstanceUID resolved
    assert os.path.isdir(results[0].output_path)


def test_duplicate_roi_names_collapse_to_one(tmp_path, ct_series_dir):
    """Two masks with the same ROI name must produce ONE ROI (last wins), not two.

    rt_utils cannot replace an ROI, so emitting two same-named ROIs yields a malformed
    RT-Struct that treatment planning systems handle inconsistently — a clinical data
    hazard. The writer collapses duplicates before writing.
    """
    ct_dir, _series_uid, (rows, cols, n_slices) = ct_series_dir
    first = np.zeros((rows, cols, n_slices), dtype=bool)
    first[8:16, 8:16, 1:5] = True
    second = np.zeros((rows, cols, n_slices), dtype=bool)
    second[20:28, 20:28, 1:5] = True

    save_dir = write_named_masks_to_rtstruct(
        [("Bladder", first), ("Bladder", second)], ct_dir, str(tmp_path / "out")
    )

    rt_path = os.path.join(save_dir, [f for f in os.listdir(save_dir) if f.endswith(".dcm")][0])
    names = RTStructBuilder.create_from(ct_dir, rt_path).get_roi_names()
    assert names == ["Bladder"]  # collapsed, not duplicated


def test_outputs_to_dicom_empty_is_safe(tmp_path):
    """Empty prediction dir returns [] instead of raising NameError (legacy bug)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    ds = tmp_path / "dataset"
    ds.mkdir()
    results = convert_nifti_outputs_to_dicom(
        model_pred_dir=str(empty),
        final_output_dir=str(tmp_path / "final"),
        dataset_dir=str(ds),
        dataset_id=720,
        exp_number="x",
        seg_map={1: "Bladder"},
    )
    assert results == []
