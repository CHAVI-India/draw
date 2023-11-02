# Adapted from https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/dicom_io.py
# Adapted from https://github.com/Sikerdebaard/dcmrtstruct2nii/tree/master/dcmrtstruct2nii
import json
import os
import os.path
import shutil
import tempfile
from csv import DictWriter
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from pydicom import dcmread

from a9t.adapters.nnunetv2 import NNUNetV2Adapter
from a9t.class_mapping import ALL_SEG_MAP
from a9t.constants import (
    TEMP_DIR_BASE,
    NNUNET_RAW_DATA_ENV_KEY,
    BASE_DIR,
    DB_NAME,
)
from a9t.converters.dcm2nii import convert_DICOM_to_Multi_NIFTI


def nnunet_preprocess(dataset_id: str, nnunet_adapter: NNUNetV2Adapter):
    nnunet_adapter.preprocess(dataset_id)


def convert_dicom_dir_to_nnunet_dataset(
    dicom_dir: str,
    dataset_id: str,
    dataset_name: str,
    sample_number: str,
    only_original:bool,
    data_tag: str = "seg",
    extension: str = "nii.gz",
) -> str:
    """
    Converts dicom_dir into nnUnet format dataset

    Args:
        dicom_dir: (str) Full path of extracted DICOM files. The dir should contains .dcm files
        from the same series. RT Struct should be present in the directory

        dataset_id: (str) 3-digit ID of the nn UNet dataset.

        dataset_name: (str) Name of the dataset. Eg: Heart, Spleen.

        sample_number: (str) sample number, Eg 009 in la_009.nii.gz

        data_tag: (str) tag of the data, Eg la in la_009.nii.gz

        extension: (str) File extension, Eg: nii.gz,

    Assumptions:
        - dataset_id is valid as per nnUNet requirements
        - NnUNet env variables are set
    """
    img_save_path, seg_save_path, dataset_dir = get_data_save_paths(
        dataset_id,
        dataset_name,
        data_tag,
        sample_number,
        extension,
    )

    seg_map = ALL_SEG_MAP[dataset_name]
    convert_dicom_to_nifti(dicom_dir, img_save_path, seg_save_path, seg_map, only_original)
    make_dataset_json_file(dataset_dir, seg_map=seg_map)
    append_data_to_db(
        dataset_id,
        sample_number,
        get_immediate_dicom_parent_dir(dicom_dir),
        dataset_dir,
    )
    return dataset_dir


def append_data_to_db(
    dataset_id,
    sample_number,
    dcm_root_dir,
    dataset_dir,
):
    """
    Adds data to CSV file for later usage
    """
    db_path = f"{dataset_dir}/{DB_NAME}"

    data = {
        "DatasetID": dataset_id,
        "SampleNumber": sample_number,
        "DICOMRootDir": dcm_root_dir,
    }

    exists = os.path.exists(db_path)

    with open(db_path, "a+") as fp:
        csv_writer = DictWriter(fp, fieldnames=data.keys())
        if exists:
            csv_writer.writerow(data)
        else:
            csv_writer.writeheader()
            csv_writer.writerow(data)



def get_data_save_paths(
    dataset_id,
    dataset_name,
    data_tag,
    sample_number,
    extension,
):
    if BASE_DIR is None:
        raise ValueError(f"Value of {NNUNET_RAW_DATA_ENV_KEY} is not set. Aborting...")

    dataset_dir = f"{BASE_DIR}/Dataset{dataset_id}_{dataset_name}"
    train_dir, labels_dir = f"{dataset_dir}/imagesTr", f"{dataset_dir}/labelsTr"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    img_save_path, seg_save_path = (
        f"{train_dir}/{data_tag}_{sample_number}_0000.{extension}",
        f"{labels_dir}/{data_tag}_{sample_number}.{extension}",
    )
    return img_save_path, seg_save_path, dataset_dir


def is_rt_file(file_name):
    f = dcmread(file_name)
    return f.Modality == "RTSTRUCT"


def get_rt_file_path(dicom_dir) -> str:
    # We expect only 1 RS per dicom series
    return [
        file_name
        for file_name in glob(f"{dicom_dir}/**/**.dcm", recursive=True)
        if is_rt_file(file_name)
    ][0]


def get_files_not_rt(dicom_dir) -> list[str]:
    return [
        file_name
        for file_name in glob(f"{dicom_dir}/**/**.dcm", recursive=True)
        if not is_rt_file(file_name)
    ]


def get_immediate_dicom_parent_dir(dicom_dir):
    one_dcm_path = glob(f"{dicom_dir}/**/**.dcm", recursive=True)[0]
    one_dcm_path = Path(one_dcm_path)
    return str(one_dcm_path.parent)


def convert_dicom_to_nifti(dicom_dir, img_save_path, seg_save_path, seg_map, only_original):
    with tempfile.TemporaryDirectory(dir=TEMP_DIR_BASE) as temp_dir:
        print("Working on", dicom_dir)
        rt_file_path = get_rt_file_path(dicom_dir)
        dicom_dir_immediate_parent = get_immediate_dicom_parent_dir(dicom_dir)
        convert_DICOM_to_Multi_NIFTI(
            rt_file_path,
            dicom_dir_immediate_parent,
            temp_dir,
            img_save_path,
            structures=list(seg_map.values()),
            mask_background_value=0,
            mask_foreground_value=1,
            convert_original_dicom=True,
            only_original=only_original,
        )
        if not only_original:
            combine_masks_to_multilabel_file(temp_dir, seg_save_path, seg_map)


def combine_masks_to_multilabel_file(masks_dir, multilabel_file, seg_map):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """
    one_mask = glob(f"{masks_dir}/**.nii.gz")[0]
    reference_image = nib.load(one_mask)
    output_image: np.ndarray = np.zeros(reference_image.shape).astype(np.uint8)

    for seg_fill_value, seg_name in seg_map.items():
        print("Processing Map: ", seg_name)
        if os.path.exists(f"{masks_dir}/{seg_name}.nii.gz"):
            img = nib.load(f"{masks_dir}/{seg_name}.nii.gz").get_fdata()
        else:
            print(f"Mask {seg_name} is missing. Filling with zeros.")
            img = np.zeros(reference_image.shape)
        output_image[img > 0.5] = seg_fill_value

    nib.save(nib.Nifti1Image(output_image, reference_image.affine), multilabel_file)


def make_dataset_json_file(dataset_dir, seg_map):
    samples = glob(f"{dataset_dir}/imagesTr/*.nii.gz")
    train_samples = int(1 * len(samples))
    test_samples = len(samples) - train_samples

    json_data = {
        "channel_names": {
            "0": "CT",
        },
        "labels": {
            "background": 0,
            **{value: key for key, value in seg_map.items()},
        },
        "numTraining": train_samples,
        "file_ending": ".nii.gz",
        "numTest": test_samples,
    }

    with open(f"{dataset_dir}/dataset.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def clear_old_training_data(dataset_id, dataset_name):
    dir_path = f"{BASE_DIR}/Dataset{dataset_id}_{dataset_name}"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
