"""Pure constants. NO I/O, NO env reads, NO side effects at import.

The legacy ``draw/config.py`` mixed real constants with import-time work
(reading ``env.draw.yml`` and scanning the model YAML dir), so importing almost
any module crashed without those files present. Keeping constants here, separate
from the runtime config loader, lets ``draw_core`` / ``draw_conversion`` be
imported anywhere — including tests with no env file and no GPU.
"""

from __future__ import annotations

# nnU-Net environment variable keys
NNUNET_RAW_DATA_ENV_KEY = "nnUNet_raw"
NNUNET_RESULTS_DATA_ENV_KEY = "nnUNet_results"
NNUNET_PREPROCESSED_ENV_KEY = "nnUNet_preprocessed"

# Filenames produced/consumed by nnU-Net
DATASET_JSON_FILENAME = "dataset.json"
PLANS_JSON_FILENAME = "plans.json"
SUMMARY_JSON_FILENAME = "summary.json"

# RT-Struct / DICOM
RT_DEFAULT_FILE_NAME = "AUTOSEGMENT.RT.dcm"
RTSTRUCT_STRING = "RTSTRUCT"
DCM_REGEX = "**/**.dcm"

# Dataset/sample conventions
DEFAULT_DATASET_TAG = "seg"
SAMPLE_SEP_DELIM = "seg_"
SAMPLE_NUMBER_ZFILL = 3
DEFAULT_MASK_NAME = "default.nii.gz"
DEFAULT_FOLD = "0"
MODEL_FOLDS = ["0", "1", "2", "3", "4"]

# Per-dataset sidecar mapping sample numbers to source DICOM dirs
DB_NAME = "db.json"

# DICOM tag tuples
DICOM_TAG_MODALITY = (0x0008, 0x0060)
DICOM_TAG_SERIES_INSTANCE_UID = (0x0020, 0x000E)
