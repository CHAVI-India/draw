import glob
import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from rt_utils import RTStructBuilder

from a9t.class_mapping import ALL_SEG_MAP
from a9t.constants import DB_NAME

CSV_FILE_PATH = "data/db/db.csv"
NNUNET_RESULTS_KEY = "nnUNet_results"


def convert_multilabel_nifti_to_rtstruct(
    nifti_file_path: str,
    dicom_dir: str,
    save_dir: str,
    label_to_name_map: dict[int, str],
    debug: bool = True,
) -> None:
    """Convert multiple NIFTI files to RT"""

    os.makedirs(save_dir, exist_ok=True)

    rt_path = f"{save_dir}/Pred_RT.dcm"

    rtstruct = None
    if os.path.exists(rt_path):
        rtstruct = RTStructBuilder.create_from(dicom_dir, rt_path)
    else:
        rtstruct = RTStructBuilder.create_new(dicom_dir)

    nifti_mask = nib.load(nifti_file_path)
    np_mask = np.asanyarray(nifti_mask.dataobj)
    np_mask = np.transpose(np_mask, [1, 0, 2])

    for idx, name in label_to_name_map.items():
        if debug:
            print("Processing Mask:", name)
        curr_mask = np_mask == idx
        rtstruct.add_roi(mask=curr_mask, name=name)

    rtstruct.save(rt_path)
    if debug:
        print("RT saved at:", save_dir)


def get_dcm_root_from_csv(dataset_id: int, sample_no: str, dataset_dir):
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
        return dcm_root_dir, os.path.split(dcm_root_dir)[-1]
    warnings.warn(f"No DICOM dir:{dataset_id}, {sample_no} not found")
    return "", ""


def get_sample_summary(s_no, summaries):
    # summaries is the array 'metric_per_case'
    s_summary = [s for s in summaries if f"seg_{s_no}" in s["reference_file"]][0]

    d = {}

    for idx in s_summary["metrics"].keys():
        d[ALL_SEG_MAP["TSGyne"][int(idx)]] = round(s_summary["metrics"][idx]["Dice"], 4)
    # metrics, 1, Dice
    return d


def modify_splits(splits: dict):
    # train, val
    d = {}
    for key in splits.keys():
        for i in splits[key]:
            d[i.split("_")[-1]] = key
    print(d)
    return d


def add_to_output_csv(dataset_id: int, summaries: list[dict], splits, save_path):
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
            _, series_id = get_dcm_root_from_csv(dataset_id, s_no)

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


def get_sample_number_from_nifti_path(nifti_path, delim="seg_"):
    print("NIFTI path", nifti_path)
    _, txt = nifti_path.split(delim)
    txt = txt.strip("_")
    # 014.nii.gz
    return txt.split(".")[0]


def convert_nifti_outputs_to_dicom(
    model_pred_dir,
    preds_dir,
    dataset_dir,
    dataset_id,
    exp_number,
    seg_map,
):
    dataset_tag = "seg"
    print("Model Pred Dir: ", model_pred_dir)
    for nifti_file_path in glob.glob(f"{model_pred_dir}/**.nii.gz"):
        sample_no = get_sample_number_from_nifti_path(nifti_file_path, dataset_tag)
        print("Processing Sample: ", sample_no)
        dcm_root_dir, dcm_parent_folder = get_dcm_root_from_csv(
            dataset_id, sample_no, dataset_dir
        )
        convert_multilabel_nifti_to_rtstruct(
            nifti_file_path=nifti_file_path,
            dicom_dir=dcm_root_dir,
            save_dir=f"{preds_dir}/{exp_number}/{dcm_parent_folder}",
            label_to_name_map=seg_map,
        )
