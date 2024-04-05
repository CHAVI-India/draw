import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
from rt_utils import RTStructBuilder

from draw.config import (
    CSV_FILE_PATH,
    DB_NAME,
    LOG,
    DEFAULT_DATASET_TAG,
    SAMPLE_SEP_DELIM,
    RT_DEFAULT_FILE_NAME,
    DicomKeyToTag,
)
from draw.dao.db import DBConnection
from draw.dao.common import Status
from draw.evaluate.evaluate import get_sample_summary
from draw.utils.ioutils import get_dicom_attribute_from_dir


def convert_multilabel_nifti_to_rtstruct(
    nifti_file_path: str,
    dicom_dir: str,
    save_dir: str,
    label_to_name_map: dict[int, str],
) -> str:
    """Convert multiple NIFTI files to RT"""

    os.makedirs(save_dir, exist_ok=True)

    rt_path = os.path.join(save_dir, RT_DEFAULT_FILE_NAME)

    rtstruct = build_rt_struct(dicom_dir, rt_path)
    np_mask = make_mask_from_rt(nifti_file_path)

    for idx, name in label_to_name_map.items():
        LOG.info(f"Processing Mask {name}")
        curr_mask = np_mask == idx
        rtstruct.add_roi(mask=curr_mask, name=name)

    rtstruct.save(rt_path)
    LOG.info(f"RT for {nifti_file_path} saved at {rt_path}")
    return save_dir


def make_mask_from_rt(nifti_file_path):
    nifti_mask = nib.load(nifti_file_path)
    np_mask = np.asanyarray(nifti_mask.dataobj)
    np_mask = np.transpose(np_mask, [1, 0, 2])
    return np_mask


def build_rt_struct(dicom_dir, rt_path):
    if os.path.exists(rt_path):
        rtstruct = RTStructBuilder.create_from(dicom_dir, rt_path)
    else:
        rtstruct = RTStructBuilder.create_new(dicom_dir)
    return rtstruct


def get_dcm_root(dataset_id: int, sample_no: str, dataset_dir):
    # Path and dir name
    df = pd.read_json(
        f"{dataset_dir}/{DB_NAME}",
        dtype={
            "DatasetID": int,
            "SampleNumber": str,
            "DICOMRootDir": str,
        },
    )
    op = df.loc[(df["DatasetID"] == dataset_id) & (df["SampleNumber"] == sample_no)]
    if not op.empty:
        dcm_root_dir = op["DICOMRootDir"].iloc[0]
        return dcm_root_dir, get_dicom_attribute_from_dir(
            dcm_root_dir, DicomKeyToTag.series_instance_uid
        )
    LOG.warning(f"No DICOM dir:{dataset_id}, Sample {sample_no} not found")
    return "", ""


def modify_splits(splits: dict):
    # train, val
    d = {}
    for key in splits.keys():
        for i in splits[key]:
            d[i.split("_")[-1]] = key
    LOG.debug(d)
    return d


def add_to_output_csv(dataset_id: int, summaries: list[dict], splits, save_path):
    # Deprecated
    # Path and dir name
    df = pd.read_csv(
        CSV_FILE_PATH,
        dtype={
            "DatasetID": int,
            "SampleNumber": str,
            "DICOMRootDir": str,
        },
    )

    df_op = pd.DataFrame()
    sample_splits = modify_splits(splits)

    all_sample_zip = list(zip(df.DatasetID, df.SampleNumber))
    for d_id, s_no in all_sample_zip:
        if d_id == dataset_id:
            _, series_id = get_dcm_root(dataset_id, s_no)

            sample_summary = get_sample_summary(s_no, summaries)
            df_op = df_op._append(
                {
                    "DICOMId": series_id,
                    "NNUNetSampleNo": s_no,
                    "Split": sample_splits[s_no],
                    **sample_summary,
                },
                ignore_index=True,
            )

    df_op.to_csv(f"{save_path}/dice.csv", index=False)


def get_sample_number_from_nifti_path(nifti_path, delim=SAMPLE_SEP_DELIM):
    LOG.debug(f"NIFTI path {nifti_path}")
    _, txt = nifti_path.split(delim)
    txt = txt.strip("_")
    # 014.nii.gz
    return txt.split(".")[0]


def convert_nifti_outputs_to_dicom(
    model_pred_dir,
    final_output_dir,
    dataset_dir,
    dataset_id,
    exp_number,
    seg_map,
):
    dataset_tag = DEFAULT_DATASET_TAG

    for nifti_file_path in glob.glob(f"{model_pred_dir}/**.nii.gz"):
        sample_no = get_sample_number_from_nifti_path(nifti_file_path, dataset_tag)
        dcm_root_dir, series_name = get_dcm_root(
            dataset_id,
            sample_no,
            dataset_dir,
        )
        save_dir = convert_multilabel_nifti_to_rtstruct(
            nifti_file_path=nifti_file_path,
            dicom_dir=dcm_root_dir,
            save_dir=f"{final_output_dir}/{exp_number}/{series_name}",
            label_to_name_map=seg_map,
        )

    LOG.info(f"Updating {series_name} to {save_dir}, {Status.PREDICTED}")

    DBConnection.update_record_by_series_name(
        series_name=series_name,
        output_path=save_dir,
        status=Status.PREDICTED,
    )
    return f"{final_output_dir}/{exp_number}"
