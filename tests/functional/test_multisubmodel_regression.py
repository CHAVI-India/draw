"""Probe: does writing multiple submodels' RT-Structs into ONE study dir keep all labels?

This mirrors the TSPrime flagship: 3 submodels (OARs / CTVp / CTVn) for the SAME study
all write to the same series-keyed output dir. The labels must accumulate into one
multi-label RT-Struct. This test exists to catch the idempotency-vs-accumulation
tension on the clinical path.
"""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
from rt_utils import RTStructBuilder

from draw_conversion.nifti2rt import convert_multilabel_nifti_to_rtstruct


def _roi_names(dicom_dir: str, rt_path: str) -> list[str]:
    return RTStructBuilder.create_from(dicom_dir, rt_path).get_roi_names()


def test_two_submodels_same_study_dir_keep_all_labels(tmp_path, ct_series_dir, multilabel_nifti):
    ct_dir, _uid, (rows, cols, n_slices) = ct_series_dir
    nifti_a, _ = multilabel_nifti  # has labels 1,2

    # A second, different single-label prediction (a separate submodel, e.g. CTVn).
    arr_b = np.zeros((rows, cols, n_slices), dtype=np.uint8)
    arr_b[4:10, 4:10, 1:5] = 1
    nifti_b = tmp_path / "seg_000_b.nii.gz"
    nib.save(nib.Nifti1Image(arr_b, np.eye(4)), str(nifti_b))

    save_dir = tmp_path / "results" / "series_X"  # SAME dir for both submodels

    convert_multilabel_nifti_to_rtstruct(nifti_a, ct_dir, str(save_dir), {1: "Bladder"})
    convert_multilabel_nifti_to_rtstruct(str(nifti_b), ct_dir, str(save_dir), {1: "CTVn"})

    dcm_files = [f for f in os.listdir(save_dir) if f.endswith(".dcm")]
    rt_path = os.path.join(str(save_dir), dcm_files[0])
    names = _roi_names(ct_dir, rt_path)

    # On the real clinical path these are different submodels of the SAME study; both
    # structures must be present in the final RT-Struct.
    assert "Bladder" in names, f"Bladder lost! ROIs={names}"
    assert "CTVn" in names, f"CTVn lost! ROIs={names}"
