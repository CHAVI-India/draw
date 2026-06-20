"""DICOM directories -> nnU-Net dataset on disk.

Ported from ``draw/preprocess/preprocess_data.py``. Changes versus the legacy code:

* Uses ``draw_conversion`` for DICOM<->NIfTI work and ``draw_conversion.dicom_io``
  for filesystem/DICOM helpers (no ``draw.utils.ioutils``).
* Persists the sample->DICOM mapping through the typed ``SampleRecord`` sidecar
  (``draw_core.sidecar.append_sidecar``) instead of writing raw dicts.
* The raw-data directory is passed in explicitly (from ``CoreConfig``) rather than
  read from ``os.getenv`` deep inside the call stack.
* The seg map / submodels come from a parsed ``ModelConfig`` (``draw_core.models``),
  not the removed ``ALL_SEG_MAP`` global.
* ``make_dataset_json_file`` no longer uses ``int(1*len(samples))``; it records all
  samples as training and zero test (see comment).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from glob import glob

import nibabel as nib
import numpy as np

from draw_conversion.dcm2nii import convert_dicom_to_multi_nifti
from draw_conversion.dicom_io import (
    get_immediate_dicom_parent_dir,
    get_rt_file_path,
    normpath,
)
from draw_core.constants import (
    DATASET_JSON_FILENAME,
    DEFAULT_DATASET_TAG,
    DEFAULT_MASK_NAME,
    SAMPLE_NUMBER_ZFILL,
)
from draw_core.logging import get_logger
from draw_core.models import ModelConfig, SubModel
from draw_core.sidecar import SampleRecord, append_sidecar

_default_log = get_logger(__name__)


def convert_dicom_dir_to_nnunet_dataset(
    dicom_dir: str,
    dataset_id: int,
    dataset_name: str,
    sample_number: str,
    seg_map: dict[int, str],
    raw_dir: str,
    data_tag: str = DEFAULT_DATASET_TAG,
    extension: str = "nii.gz",
    only_original: bool = True,
    logger: logging.Logger | None = None,
) -> str:
    """Convert one DICOM directory into an nnU-Net dataset sample.

    Returns the dataset directory. Assumes ``dataset_id`` is valid per nnU-Net.
    """
    log = logger or _default_log
    img_save_path, seg_save_path, dataset_dir = get_data_save_paths(
        dataset_id, dataset_name, data_tag, sample_number, extension, raw_dir
    )

    convert_dicom_to_nifti(
        dicom_dir, img_save_path, seg_save_path, seg_map, only_original, log
    )
    make_dataset_json_file(dataset_dir, seg_map=seg_map, modality="CT")
    append_sidecar(
        dataset_dir,
        SampleRecord(
            dataset_id=int(dataset_id),
            sample_number=str(sample_number),
            dicom_root_dir=get_immediate_dicom_parent_dir(dicom_dir),
        ),
    )
    return dataset_dir


def get_data_save_paths(
    dataset_id: int,
    dataset_name: str,
    data_tag: str,
    sample_number: str,
    extension: str,
    raw_dir: str,
) -> tuple[str, str, str]:
    dataset_dir = normpath(f"{raw_dir}/Dataset{dataset_id}_{dataset_name}")
    train_dir = normpath(f"{dataset_dir}/imagesTr")
    labels_dir = normpath(f"{dataset_dir}/labelsTr")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    img_save_path = normpath(f"{train_dir}/{data_tag}_{sample_number}_0000.{extension}")
    seg_save_path = normpath(f"{labels_dir}/{data_tag}_{sample_number}.{extension}")
    return img_save_path, seg_save_path, dataset_dir


def convert_dicom_to_nifti(
    dicom_dir: str,
    img_save_path: str,
    seg_save_path: str,
    seg_map: dict[int, str],
    only_original: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    log = logger or _default_log
    with tempfile.TemporaryDirectory() as temp_dir:
        rt_file_path = get_rt_file_path(dicom_dir) if not only_original else None
        dicom_dir_immediate_parent = get_immediate_dicom_parent_dir(dicom_dir)
        convert_dicom_to_multi_nifti(
            rt_file_path,
            dicom_dir_immediate_parent,
            temp_dir,
            img_save_path,
            structures=list(seg_map.values()),
            mask_background_value=0,
            mask_foreground_value=1,
            only_original=only_original,
        )
        if not only_original:
            combine_masks_to_multilabel_file(temp_dir, seg_save_path, seg_map, log)


def combine_masks_to_multilabel_file(
    masks_dir: str,
    output_nifti_path: str,
    seg_map: dict[int, str],
    logger: logging.Logger | None = None,
) -> None:
    """Merge the per-class binary masks into a single multilabel NIfTI for training."""
    log = logger or _default_log
    one_mask = glob(f"{masks_dir}/{DEFAULT_MASK_NAME}")[0]
    reference_image = nib.load(one_mask)
    output_image: np.ndarray = np.zeros(reference_image.shape).astype(np.uint8)

    for seg_fill_value, seg_name in seg_map.items():
        log.info("Processing Map %s for %s", seg_name, output_nifti_path)
        mask_path = f"{masks_dir}/{seg_name}.nii.gz"
        if os.path.exists(mask_path):
            img = nib.load(mask_path).get_fdata()
        else:
            log.warning(
                "%s missing for %s. Generated zero filled mask", seg_name, output_nifti_path
            )
            img = np.zeros(reference_image.shape)
        output_image[img > 0.5] = seg_fill_value

    nib.save(nib.Nifti1Image(output_image, reference_image.affine), output_nifti_path)


def make_dataset_json_file(dataset_dir: str, seg_map: dict[int, str], modality: str) -> None:
    samples = glob(normpath(f"{dataset_dir}/imagesTr/**.nii.gz"))
    # Every converted sample is training data; nnU-Net does its own train/val split
    # via folds, so there is no held-out test set here (legacy used int(1*len)).
    num_training = len(samples)
    num_test = 0

    json_data = {
        # TODO: Get modality from DICOM image rather than hardcoding.
        "channel_names": {"0": modality},
        "labels": {
            "background": 0,
            **{value: key for key, value in seg_map.items()},
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "numTest": num_test,
    }

    with open(normpath(f"{dataset_dir}/{DATASET_JSON_FILENAME}"), "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def run_pre_processing(
    dataset_id: int,
    model: ModelConfig,
    only_original: bool,
    parent_root_dir: str,
    sample_numbering_start: int,
    raw_dir: str,
    logger: logging.Logger | None = None,
) -> None:
    """Convert every DICOM directory under ``parent_root_dir`` for one submodel.

    Args:
        dataset_id: 3-digit nnU-Net dataset id (selects the submodel of ``model``).
        model: parsed site ``ModelConfig`` (replaces the ALL_SEG_MAP lookup).
        only_original: when True, RT-Struct files are not parsed.
        parent_root_dir: dir containing the per-study DICOM directories.
        sample_numbering_start: continue numbering samples from this index.
        raw_dir: nnU-Net raw data dir (from CoreConfig).
    """
    log = logger or _default_log
    submodel: SubModel = model.submodels[int(dataset_id)]
    all_dicom_dirs = [f.path for f in os.scandir(parent_root_dir) if f.is_dir()]
    log.info("Processing ID %s", dataset_id)
    log.info("Found %d directories to work on...", len(all_dicom_dirs))

    for idx, dicom_dir in enumerate(all_dicom_dirs, start=sample_numbering_start):
        sample_number = str(idx).zfill(SAMPLE_NUMBER_ZFILL)
        convert_dicom_dir_to_nnunet_dataset(
            dicom_dir,
            dataset_id,
            submodel.name,
            sample_number,
            submodel.seg_map,
            raw_dir=raw_dir,
            data_tag=DEFAULT_DATASET_TAG,
            extension="nii.gz",
            only_original=only_original,
            logger=log,
        )
