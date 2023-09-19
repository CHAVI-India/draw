import argparse
import os

from components.preprocess.preprocess_data import convert_dicom_dir_to_nnunet_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert DICOM files to NIFTI for processing...")

    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--sample_start", type=int, required=False, default=0)

    args = parser.parse_args()

    raw_data_dir = args.root_dir
    dataset_id = args.dataset_id
    dataset_name = args.dataset_name
    sample_start = args.sample_start

    all_dicom_dirs = [f.path for f in os.scandir(raw_data_dir) if f.is_dir()]

    for idx, dicom_dir in enumerate(all_dicom_dirs, start=sample_start):
        sample_number = str(idx).zfill(3)

        convert_dicom_dir_to_nnunet_dataset(
            dicom_dir,
            dataset_id,
            dataset_name,
            sample_number,
            data_tag="seg",
            extension="nii.gz",
        )
