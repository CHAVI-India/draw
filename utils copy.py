# Adapted from https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/dicom_io.py
# Adapted from https://github.com/Sikerdebaard/dcmrtstruct2nii/tree/master/dcmrtstruct2nii
import logging
import os
import os.path
import shutil
import tempfile
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from dcmrtstruct2nii.adapters.convert.filenameconverter import FilenameConverter
from dcmrtstruct2nii.adapters.convert.rtstructcontour2mask import DcmPatientCoords2Mask
from dcmrtstruct2nii.adapters.input.contours.rtstructinputadapter import (
    RtStructInputAdapter,
)
from dcmrtstruct2nii.adapters.input.image.dcminputadapter import DcmInputAdapter
from dcmrtstruct2nii.adapters.output.niioutputadapter import NiiOutputAdapter
from dcmrtstruct2nii.exceptions import (
    ContourOutOfBoundsException,
    PathDoesNotExistException,
)
from pydicom import dcmread

from class_mapping import seg_map

TEMP_DIR_BASE = "temp"
OUTPUT_DIR = "output"
NNUNET_RAW_DATA_ENV_KEY = "nnUNet_raw"


def convert_dicom_dir_to_nnunet_dataset(
    dicom_dir: str,
    dataset_id: str,
    dataset_name: str,
    sample_number: str,
    data_tag: str = "cus",
    extension: str = "nii.gz",
) -> None:
    """
    Converts dicom_dir into nnUnet format dataset

    Args:
        dicom_dir: (str) Full path of extracted DICOM files. The dir should contains .dcm files
        from the same series. RT Struct should be present in the directory

        dataset_id: (str) 3 digit ID of the nn UNet dataset.

        datatset_name: (str) Name of the dataset. Eg: Heart, Spleen.

        sample_number: (str) sample number, Eg 009 in la_009.nii.gz

        data_tag: (str) tag of the data, Eg la inla_009.nii.gz

        extension: (str) File extension, Eg: nii.gz,

    Assumptions:
        - dataset_id is valid as per nnUNet requirements
        - NnUNet env variables are set
    """
    img_save_path, seg_save_path = get_data_save_paths(
        dataset_id,
        dataset_name,
        data_tag,
        sample_number,
        extension,
    )

    convert_dicom_image_to_nifti(dicom_dir, img_save_path)
    convert_dicom_labels_to_nifti(dicom_dir, seg_save_path)


def get_data_save_paths(
    dataset_id,
    dataset_name,
    data_tag,
    sample_number,
    extension,
):
    BASE_DIR = os.environ.get(NNUNET_RAW_DATA_ENV_KEY, None)

    if BASE_DIR is None:
        raise ValueError(f"Value of {NNUNET_RAW_DATA_ENV_KEY} is not set. Aborting...")

    dataset_dir = f"{BASE_DIR}/Dataset{dataset_id}_{dataset_name}"
    train_dir, labels_dir = f"{dataset_dir}/imagesTr", f"{dataset_dir}/labelsTr"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    img_save_path, seg_save_path = (
        f"{train_dir}/{data_tag}_{sample_number}.{extension}",
        f"{labels_dir}/{data_tag}_{sample_number}.{extension}",
    )
    return img_save_path, seg_save_path


def convert_dicom_image_to_nifti(dicom_dir, image_save_path):
    with tempfile.TemporaryDirectory(dir=TEMP_DIR_BASE) as temp_dir:
        DCM2NII_SUBPROCESS_CALL = f"dcm2niix -o {temp_dir} -z y {dicom_dir}"
        os.system(DCM2NII_SUBPROCESS_CALL)  # TODO: Use subprocess module if possible
        nifti_name = glob(f"{temp_dir}/**.nii.gz")[0]
        shutil.copy(nifti_name, image_save_path)


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


def get_immediate_dicom_parent_dir(dicom_dir):
    one_dcm_path = glob(f"{dicom_dir}/**/**.dcm")[0]
    one_dcm_path = Path(one_dcm_path)
    return str(one_dcm_path.parent)


def convert_dicom_labels_to_nifti(dicom_dir, seg_save_path):
    with tempfile.TemporaryDirectory(dir=TEMP_DIR_BASE) as temp_dir:
        rt_file_path = get_rt_file_path(dicom_dir)
        dicom_dir_immediate_parent = get_immediate_dicom_parent_dir(dicom_dir)
        dcmrtstruct2nii(
            rt_file_path,
            dicom_dir_immediate_parent,
            temp_dir,
            mask_background_value=0,
            convert_original_dicom=False,
        )
        combine_masks_to_multilabel_file(temp_dir, seg_save_path)


def combine_masks_to_multilabel_file(masks_dir, multilabel_file):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """
    one_mask = glob(f"{masks_dir}/**.nii.gz")[0]
    ref_img = nib.load(one_mask)
    img_out = np.zeros(ref_img.shape).astype(np.uint8)

    for seg_value, mask_name in seg_map.items():
        if os.path.exists(f"{masks_dir}/{mask_name}.nii.gz"):
            img = nib.load(f"{masks_dir}/{mask_name}.nii.gz").get_fdata()
        else:
            print(f"Mask {mask_name} is missing. Filling with zeros.")
            img = np.zeros(ref_img.shape)
        img_out[img > 0.5] = seg_value + 1

    nib.save(nib.Nifti1Image(img_out, ref_img.affine), multilabel_file)


def dcmrtstruct2nii(
    rtstruct_file,
    dicom_file,
    output_path,
    structures=None,
    gzip=True,
    mask_background_value=0,
    mask_foreground_value=255,
    convert_original_dicom=True,
    series_id=None,
):  # noqa: C901 E501
    """
    Converts A DICOM and DICOM RT Struct file to nii

    :param rtstruct_file: Path to the rtstruct file
    :param dicom_file: Path to the dicom file
    :param output_path: Output path where the masks are written to
    :param structures: Optional, list of structures to convert
    :param gzip: Optional, output .nii.gz if set to True, default: True
    :param series_id: Optional, the Series Instance UID. Use  to specify the ID corresponding to the image if there are
    dicoms from more than one series in `dicom_file` folder

    :raise InvalidFileFormatException: Raised when an invalid file format is given.
    :raise PathDoesNotExistException: Raised when the given path does not exist.
    :raise UnsupportedTypeException: Raised when conversion is not supported.
    :raise ValueError: Raised when mask_background_value or mask_foreground_value is invalid.
    """
    output_path = os.path.join(output_path, "")  # make sure trailing slash is there

    if not os.path.exists(rtstruct_file):
        raise PathDoesNotExistException(
            f"rtstruct path does not exist: {rtstruct_file}"
        )

    if not os.path.exists(dicom_file):
        raise PathDoesNotExistException(f"DICOM path does not exists: {dicom_file}")

    if mask_background_value < 0 or mask_background_value > 255:
        raise ValueError(
            f"Invalid value for mask_background_value: {mask_background_value}, must be between 0 and 255"
        )

    if mask_foreground_value < 0 or mask_foreground_value > 255:
        raise ValueError(
            f"Invalid value for mask_foreground_value: {mask_foreground_value}, must be between 0 and 255"
        )

    if structures is None:
        structures = []

    os.makedirs(output_path, exist_ok=True)

    filename_converter = FilenameConverter()
    rtreader = RtStructInputAdapter()

    rtstructs = rtreader.ingest(rtstruct_file)
    dicom_image = DcmInputAdapter().ingest(dicom_file, series_id=series_id)

    dcm_patient_coords_to_mask = DcmPatientCoords2Mask()
    nii_output_adapter = NiiOutputAdapter()
    for rtstruct in rtstructs:
        if len(structures) == 0 or rtstruct["name"] in structures:
            if "sequence" not in rtstruct:
                logging.info(
                    "Skipping mask {} no shape/polygon found".format(rtstruct["name"])
                )
                continue

            logging.info("Working on mask {}".format(rtstruct["name"]))
            try:
                mask = dcm_patient_coords_to_mask.convert(
                    rtstruct["sequence"],
                    dicom_image,
                    mask_background_value,
                    mask_foreground_value,
                )
            except ContourOutOfBoundsException:
                logging.warning(
                    f'Structure {rtstruct["name"]} is out of bounds, ignoring contour!'
                )
                continue

            mask.CopyInformation(dicom_image)

            final_file_name = f'{rtstruct["name"]}'
            final_file_name = final_file_name.lower()

            mask_filename = filename_converter.convert(final_file_name)
            nii_output_adapter.write(mask, f"{output_path}{mask_filename}", gzip)

    if convert_original_dicom:
        logging.info("Converting original DICOM to nii")
        nii_output_adapter.write(dicom_image, f"{output_path}image", gzip)

    logging.info("Success!")


if __name__ == "__main__":
    raw_data_dir = "data/raw"
    dataset_id = "700"
    dataset_name = "MultiSeg"

    all_dicom_dirs = [f.path for f in os.scandir(raw_data_dir) if f.is_dir()]

    for idx, dicom_dir in enumerate(all_dicom_dirs):
        sample_number = str(idx).zfill(4)

        convert_dicom_dir_to_nnunet_dataset(
            dicom_dir,
            dataset_id,
            dataset_name,
            sample_number,
            data_tag="seg",
            extension="nii.gz",
        )
