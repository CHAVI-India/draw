# TODO: Generalise this


from rt_utils import RTStructBuilder
import nibabel as nib
import numpy as np
from components.class_mapping import ALL_SEG_MAP
from components.preprocess.preprocess_data import get_files_not_rt
import os
import shutil
import pandas as pd

CSV_FILE_PATH = "data/db/db.csv"
NNUNET_RESULTS_KEY = "nnUNet_results"


def copy_all_not_rt_files(dicom_dir: str, save_dir: str) -> None:
    all_dcm_files = get_files_not_rt(dicom_dir)
    for src in all_dcm_files:
        _, dst_filename = os.path.split(src)
        shutil.copy(src, f"{save_dir}/{dst_filename}")


def convert_multilabel_nifti_to_rtstruct(
    nifti_file_path: str,
    dicom_dir: str,
    save_dir: str,
    label_to_name_map: dict[int, str],
    debug: bool = True,
) -> None:
    """Convert multiple NIFTI files to RT"""

    os.makedirs(save_dir, exist_ok=True)

    # copy_all_not_rt_files(dicom_dir, save_dir)

    rtstruct = RTStructBuilder.create_new(dicom_dir)
    nifti_mask = nib.load(nifti_file_path)
    np_mask = np.asanyarray(nifti_mask.dataobj)
    np_mask = np.transpose(np_mask, [1, 0, 2])

    for idx, name in label_to_name_map.items():
        if debug:
            print("Processing Mask:", name)
        curr_mask = np_mask == idx
        rtstruct.add_roi(mask=curr_mask, name=name)

    rtstruct.save(f"{save_dir}/Pred_RT.dcm")
    if debug:
        print("RT saved at:", save_dir)


def get_dicom_root_dir(dataset_id: int, sample_no: str) -> tuple[str, str]:
    # Path and dir name
    df = pd.read_csv(
        CSV_FILE_PATH,
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
    return None, None

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
            _, series_id = get_dicom_root_dir(dataset_id, s_no)

            sample_summary = get_sample_summary(s_no, summaries)
            df_op = df_op._append({
                "DICOMId": series_id,
                "NNUNetSampleNo": s_no,
                "Split": sample_splits[s_no],
                **sample_summary
            }, ignore_index=True)

    df_op.to_csv(f"{save_path}/dice.csv", index=False)








