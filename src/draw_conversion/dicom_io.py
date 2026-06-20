"""Filesystem + DICOM read helpers used by the conversion layer.

Extracted from the legacy ``draw/utils/ioutils.py``. Pure I/O, no config, no DB.
Includes the cheap wins from benchmarking: read only the tags we need
(``stop_before_pixels``/``specific_tags``) and discover RT vs image files in a
single directory scan instead of two.
"""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path

from pydicom import dcmread

from draw_core.constants import DCM_REGEX, DICOM_TAG_SERIES_INSTANCE_UID, RTSTRUCT_STRING


def normpath(path: str) -> str:
    return os.path.normpath(path)


def list_dcm_files(dicom_dir: str) -> list[str]:
    return glob(normpath(f"{dicom_dir}/{DCM_REGEX}"), recursive=True)


def _modality(file_name: str) -> str | None:
    """Read just the Modality tag without loading pixel data."""
    ds = dcmread(file_name, stop_before_pixels=True, specific_tags=["Modality"])
    return getattr(ds, "Modality", None)


def split_rt_and_image_files(dicom_dir: str) -> tuple[list[str], list[str]]:
    """Single scan -> (rt_struct_files, image_files). Replaces two glob+read passes."""
    rt_files: list[str] = []
    image_files: list[str] = []
    for f in list_dcm_files(dicom_dir):
        if _modality(f) == RTSTRUCT_STRING:
            rt_files.append(f)
        else:
            image_files.append(f)
    return rt_files, image_files


def get_rt_file_path(dicom_dir: str) -> str:
    """Path to the single expected RT-Struct file in a series."""
    rt_files, _ = split_rt_and_image_files(dicom_dir)
    if not rt_files:
        raise FileNotFoundError(f"No RTSTRUCT file found in {dicom_dir}")
    return rt_files[0]


def get_one_dcm_path(dicom_dir: str) -> Path:
    files = list_dcm_files(dicom_dir)
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    return Path(files[0])


def get_immediate_dicom_parent_dir(dicom_dir: str) -> str:
    return str(get_one_dcm_path(dicom_dir).parent)


def get_series_instance_uid(dicom_dir: str) -> str | None:
    """SeriesInstanceUID from any one file in the dir (tag read only)."""
    ds = dcmread(
        str(get_one_dcm_path(dicom_dir)),
        stop_before_pixels=True,
        specific_tags=["SeriesInstanceUID"],
    )
    if DICOM_TAG_SERIES_INSTANCE_UID in ds:
        return str(ds[DICOM_TAG_SERIES_INSTANCE_UID].value)
    return None
