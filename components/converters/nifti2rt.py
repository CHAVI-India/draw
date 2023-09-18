from rt_utils import RTStructBuilder
import nibabel as nib
import numpy as np
from components.class_mapping import ts_prime_map
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


def get_dicom_root_dir(dataset_id: int, sample_no: str) -> str:
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
        return op["DICOMRootDir"].iloc[0]
    return None


if __name__ == "__main__":
    RESULTS_DIR = os.environ.get(NNUNET_RESULTS_KEY, None)

    if RESULTS_DIR is None:
        raise ValueError(f"Value of {NNUNET_RESULTS_KEY} is not set. Aborting...")

    # TODO: get details from CLI. Also number of samples
    dataset_id = 720
    dataset_name = "TSPrime"
    dataset_tag = "seg"

    number_of_samples = 24
    for sample in range(number_of_samples):
        sample_no = str(sample).zfill(3)
        print("Processing Sample: ", sample_no)
        convert_multilabel_nifti_to_rtstruct(
            nifti_file_path=f"{RESULTS_DIR}/Dataset{dataset_id}_{dataset_name}/imagesTr_predhighres/{dataset_tag}_{sample_no}.nii.gz",
            dicom_dir=get_dicom_root_dir(dataset_id, sample_no),
            save_dir=f"{RESULTS_DIR}/Dataset{dataset_id}_{dataset_name}/nnUNetTrainer__nnUNetPlans__3d_fullres/preds/{sample_no}",
            label_to_name_map=ts_prime_map,
        )
