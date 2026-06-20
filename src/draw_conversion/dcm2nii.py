"""DICOM (+ optional RT-Struct) -> NIfTI.

Refactored from ``draw/utils/dcm2nii.py``. Depends only on the conversion stack
(SimpleITK, dcmrtstruct2nii) plus a module logger — no ``draw.config`` import.
"""

from __future__ import annotations

import os

import numpy as np
import SimpleITK as sitk
from dcmrtstruct2nii.adapters.convert.rtstructcontour2mask import DcmPatientCoords2Mask
from dcmrtstruct2nii.adapters.input.contours.rtstructinputadapter import RtStructInputAdapter
from dcmrtstruct2nii.adapters.input.image.dcminputadapter import DcmInputAdapter
from dcmrtstruct2nii.adapters.output.niioutputadapter import NiiOutputAdapter
from dcmrtstruct2nii.exceptions import ContourOutOfBoundsException, PathDoesNotExistException

from draw_conversion._gzip_fast import enable_fast_gzip
from draw_core.constants import DEFAULT_MASK_NAME
from draw_core.logging import get_logger

log = get_logger(__name__)

enable_fast_gzip()


def convert_dicom_to_multi_nifti(
    rt_struct_file_path: str | None,
    dicom_file_path: str,
    output_dir: str,
    dicom_image_save_path: str,
    structures: list[str] | None = None,
    gzip: bool = True,
    mask_background_value: int = 0,
    mask_foreground_value: int = 255,
    series_id: str | None = None,
    only_original: bool = True,
    save_default_empty: bool = True,
) -> None:
    """Convert a DICOM image (and optionally its RT-Struct masks) to NIfTI."""
    output_dir = os.path.join(output_dir, "")

    if not os.path.exists(dicom_file_path):
        raise PathDoesNotExistException(f"DICOM path does not exist: {dicom_file_path}")

    dicom_image = DcmInputAdapter().ingest(dicom_file_path, series_id=series_id)
    nii_output_adapter = NiiOutputAdapter()

    if not only_original:
        _write_rt_masks(
            rt_struct_file_path,
            dicom_image,
            output_dir,
            nii_output_adapter,
            structures or [],
            mask_background_value,
            mask_foreground_value,
            gzip,
        )

    log.info("Converting DICOM scan to NIfTI for %s", dicom_file_path)
    nii_output_adapter.write(dicom_image, dicom_image_save_path, gzip)

    if save_default_empty:
        empty = sitk.GetImageFromArray(np.zeros(sitk.GetArrayViewFromImage(dicom_image).shape,
                                                dtype=np.uint8))
        empty.CopyInformation(dicom_image)
        default_path = f"{output_dir}{DEFAULT_MASK_NAME}"
        nii_output_adapter.write(empty, default_path, gzip)
        log.info("Saved default empty mask at %s", default_path)

    log.info("Conversion for %s complete", dicom_file_path)


def _write_rt_masks(
    rt_struct_file_path: str | None,
    dicom_image,
    output_dir: str,
    nii_output_adapter: NiiOutputAdapter,
    structures: list[str],
    mask_background_value: int,
    mask_foreground_value: int,
    gzip: bool,
) -> None:
    if not rt_struct_file_path or not os.path.exists(rt_struct_file_path):
        raise PathDoesNotExistException(f"rtstruct path does not exist: {rt_struct_file_path}")
    for v in (mask_background_value, mask_foreground_value):
        if v < 0 or v > 255:
            raise ValueError(f"mask value {v} must be between 0 and 255")

    os.makedirs(output_dir, exist_ok=True)
    all_rt_structs = RtStructInputAdapter().ingest(rt_struct_file_path)
    coords_to_mask = DcmPatientCoords2Mask()

    for rtstruct in all_rt_structs:
        if structures and rtstruct["name"] not in structures:
            continue
        if "sequence" not in rtstruct:
            log.info("Skipping mask %s: no shape/polygon found", rtstruct["name"])
            continue
        log.info("Working on mask %s", rtstruct["name"])
        try:
            mask = coords_to_mask.convert(
                rtstruct["sequence"], dicom_image, mask_background_value, mask_foreground_value
            )
        except ContourOutOfBoundsException:
            log.info("Structure %s out of bounds, ignoring", rtstruct["name"])
            continue
        mask.CopyInformation(dicom_image)
        nii_output_adapter.write(mask, f"{output_dir}{rtstruct['name']}", gzip)
