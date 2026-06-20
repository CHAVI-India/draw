"""Re-running a study's RT-Struct conversion is idempotent (effectively-once).

If the pipeline is killed after writing an RT-Struct but before recording success,
the reaper re-queues the study and it is converted again. That retry must overwrite
its own output with an equivalent file — not produce a second file, and not stack
duplicate ROIs onto the first. These tests prove that.
"""

from __future__ import annotations

import os

from rt_utils import RTStructBuilder

from draw_conversion.nifti2rt import convert_multilabel_nifti_to_rtstruct


def _roi_names(dicom_dir: str, rt_path: str) -> list[str]:
    return RTStructBuilder.create_from(dicom_dir, rt_path).get_roi_names()


def test_rerunning_conversion_keeps_single_output_with_same_rois(
    tmp_path, ct_series_dir, multilabel_nifti
):
    ct_dir, _series_uid, _ = ct_series_dir
    nifti_path, seg_map = multilabel_nifti
    save_dir = tmp_path / "out"

    # Arrange + Act: convert the same study twice into the same deterministic dir.
    convert_multilabel_nifti_to_rtstruct(nifti_path, ct_dir, str(save_dir), seg_map)
    first_files = sorted(os.listdir(save_dir))

    convert_multilabel_nifti_to_rtstruct(nifti_path, ct_dir, str(save_dir), seg_map)
    second_files = sorted(os.listdir(save_dir))

    # Assert: exactly one RT-Struct, no leftover temp file, ROI set unchanged (not
    # doubled) — the retry overwrote rather than appended.
    rt_path = os.path.join(str(save_dir), first_files[0])
    assert first_files == second_files
    assert len([f for f in second_files if f.endswith(".dcm")]) == 1
    assert not any(f.endswith(".tmp") for f in second_files)

    roi_names = _roi_names(ct_dir, rt_path)
    assert sorted(roi_names) == sorted(seg_map.values())
    assert len(roi_names) == len(seg_map)  # ROIs not stacked across the two runs
