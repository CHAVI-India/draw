"""NIfTI segmentation masks -> DICOM RT-Struct.

Refactored from ``draw/utils/nifti2rt.py``. The critical change: this module no
longer writes to the database. The legacy ``convert_nifti_outputs_to_dicom``
called ``DBConnection.update_record_by_series_name(...)`` directly — a conversion
module reaching into the DAO. Here it simply RETURNS the produced
``SeriesResult`` objects and lets the caller decide what to record (DIP).

Also fixes the legacy bug where the DB update sat *outside* the per-file loop, so
only the last series was ever recorded (and an empty glob raised NameError).
"""

from __future__ import annotations

import glob
import os

import nibabel as nib
import numpy as np
from rt_utils import RTStructBuilder

from draw_contracts.dto import SeriesResult
from draw_conversion._gzip_fast import enable_fast_gzip
from draw_conversion.dicom_io import get_series_instance_uid
from draw_core.constants import (
    DEFAULT_DATASET_TAG,
    RT_DEFAULT_FILE_NAME,
    SAMPLE_SEP_DELIM,
)
from draw_core.logging import get_logger
from draw_core.sidecar import find_sample, load_sidecar

log = get_logger(__name__)

enable_fast_gzip()


def make_mask_from_rt(nifti_file_path: str) -> np.ndarray:
    """Load a multilabel NIfTI mask, oriented to match rt_utils' expectation."""
    nifti_mask = nib.load(nifti_file_path)
    np_mask = np.asanyarray(nifti_mask.dataobj)
    return np.transpose(np_mask, [1, 0, 2])


def convert_multilabel_nifti_to_rtstruct(
    nifti_file_path: str,
    dicom_dir: str,
    save_dir: str,
    label_to_name_map: dict[int, str],
) -> str:
    """Write one multilabel NIfTI as a DICOM RT-Struct under ``save_dir``.

    Idempotent: the output path is deterministic (``save_dir`` is keyed by
    SeriesInstanceUID upstream) and we always build a FRESH RT-Struct, write it to a
    temp file, then atomically ``os.replace`` it into place. So re-running the same
    study (e.g. after a crash + reaper retry) overwrites its own output with an
    equivalent file instead of appending duplicate ROIs to a stale one. This is what
    makes at-least-once processing effectively-once.
    """
    os.makedirs(save_dir, exist_ok=True)
    rt_path = os.path.join(save_dir, RT_DEFAULT_FILE_NAME)
    # rt_utils appends ".dcm" if the path lacks it, so keep the temp name ending in
    # ".dcm" (".tmp.dcm") to control the exact filename it writes.
    tmp_path = f"{rt_path}.tmp.dcm"

    # Always create_new (never create_from an existing RT) so a retry does not stack
    # ROIs on top of a previous run's output.
    rtstruct = RTStructBuilder.create_new(dicom_dir)
    np_mask = make_mask_from_rt(nifti_file_path)

    for idx, name in label_to_name_map.items():
        log.info("Processing mask %s", name)
        rtstruct.add_roi(mask=(np_mask == idx), name=name)

    rtstruct.save(tmp_path)
    os.replace(tmp_path, rt_path)  # atomic on POSIX; overwrites any prior output
    log.info("RT-Struct for %s saved at %s", nifti_file_path, rt_path)
    return save_dir


def get_sample_number_from_nifti_path(nifti_path: str, delim: str = SAMPLE_SEP_DELIM) -> str:
    # Split the basename only: the full path may contain the delimiter (e.g. a tmp
    # dir literally named ``...segment...``), which made the legacy whole-path split
    # raise "too many values to unpack".
    _, txt = os.path.basename(nifti_path).split(delim)
    return txt.strip("_").split(".")[0]


def get_dcm_root(dataset_id: int, sample_no: str, dataset_dir: str) -> tuple[str, str]:
    """Resolve the source DICOM dir + SeriesInstanceUID for a sample from the sidecar.

    Uses the typed ``SampleRecord`` sidecar model (no raw-dict string indexing).
    """
    record = find_sample(load_sidecar(dataset_dir), dataset_id, sample_no)
    if record is None:
        log.warning("No DICOM dir: dataset %s sample %s", dataset_id, sample_no)
        return "", ""
    return record.dicom_root_dir, get_series_instance_uid(record.dicom_root_dir) or ""


def convert_nifti_outputs_to_dicom(
    model_pred_dir: str,
    final_output_dir: str,
    dataset_dir: str,
    dataset_id: int,
    exp_number: str,
    seg_map: dict[int, str],
) -> list[SeriesResult]:
    """Convert every predicted NIfTI in ``model_pred_dir`` to an RT-Struct.

    Returns one ``SeriesResult`` per produced series. Does NOT touch any DB — the
    caller records status. Fixes the legacy single-result / NameError bug.
    """
    results: list[SeriesResult] = []
    for nifti_file_path in glob.glob(f"{model_pred_dir}/**.nii.gz"):
        sample_no = get_sample_number_from_nifti_path(nifti_file_path, DEFAULT_DATASET_TAG)
        dcm_root_dir, series_name = get_dcm_root(dataset_id, sample_no, dataset_dir)
        if not dcm_root_dir:
            continue
        save_dir = convert_multilabel_nifti_to_rtstruct(
            nifti_file_path=nifti_file_path,
            dicom_dir=dcm_root_dir,
            save_dir=f"{final_output_dir}/{exp_number}/{series_name}",
            label_to_name_map=seg_map,
        )
        log.info("Produced RT-Struct for series %s at %s", series_name, save_dir)
        results.append(SeriesResult(series_name=series_name, output_path=save_dir))
    return results
