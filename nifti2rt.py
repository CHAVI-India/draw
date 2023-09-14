from rt_utils import RTStructBuilder
import nibabel as nib
import numpy as np
from class_mapping import ts_prime_map
from preprocess_data import get_files_not_rt
import os
import shutil


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

    copy_all_not_rt_files(dicom_dir, save_dir)

    rtstruct = RTStructBuilder.create_new(dicom_dir)
    nifti_mask = nib.load(nifti_file_path)
    np_mask = np.asanyarray(nifti_mask.dataobj)

    for idx, name in label_to_name_map.items():
        if debug:
            print("Processing Mask:", name)
        curr_mask = np_mask == idx
        rtstruct.add_roi(mask=curr_mask, name=name)

    rtstruct.save(f"{save_dir}/RT.dcm")
    if debug:
        print("RT saved at:", save_dir)


if __name__ == "__main__":
    convert_multilabel_nifti_to_rtstruct(
        nifti_file_path="data/nnUNet_raw/Dataset720_TSPrime/labelsTr/seg_000.nii.gz",
        dicom_dir="data/raw/TS_Prime/2013030109202300012/2013030109202300012.0.1693545750295",
        save_dir="data/nnUNet_results/Dataset720_TSPrime/nnUNetTrainer__nnUNetPlans__3d_fullres/rt_000",
        label_to_name_map=ts_prime_map,
    )
