"""Shared fixtures: synthetic DICOM CT series + multilabel NIfTI prediction.

Builds a tiny but geometrically valid CT series and a matching multilabel mask so
the conversion layer can be exercised end-to-end with no GPU and no real data.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def ct_series_dir(tmp_path):
    """Write a small valid CT series to disk; return (dir, series_uid)."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    rows = cols = 32
    n_slices = 6
    series_uid = generate_uid()
    frame_uid = generate_uid()
    study_uid = generate_uid()
    d = tmp_path / "ct"
    d.mkdir()
    for i in range(n_slices):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.Modality = "CT"
        ds.Rows, ds.Columns = rows, cols
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 3.0
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i * 3)]
        ds.PixelData = np.zeros((rows, cols), dtype=np.int16).tobytes()
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.InstanceNumber = i + 1
        ds.StudyDate = "20240101"
        ds.SeriesDate = "20240101"
        ds.StudyTime = "120000"
        ds.SeriesTime = "120000"
        ds.StudyID = "1"
        ds.SeriesNumber = 1
        ds.PatientName = "TEST^FUNCTIONAL"
        ds.PatientID = "TEST001"
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.save_as(str(d / f"{i:03d}.dcm"), enforce_file_format=True)
    return str(d), series_uid, (rows, cols, n_slices)


@pytest.fixture
def multilabel_nifti(tmp_path, ct_series_dir):
    """A 2-label NIfTI mask matching the CT geometry; return its path + seg_map."""
    import nibabel as nib

    _, _, (rows, cols, n_slices) = ct_series_dir
    arr = np.zeros((rows, cols, n_slices), dtype=np.uint8)
    arr[8:16, 8:16, 1:5] = 1  # label 1 blob
    arr[20:28, 20:28, 1:5] = 2  # label 2 blob
    path = tmp_path / "seg_000.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))
    return str(path), {1: "Bladder", 2: "Anorectum"}
