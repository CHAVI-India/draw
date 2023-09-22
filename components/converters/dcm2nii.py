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


import os
import os.path


def convert_DICOM_to_Multi_NIFTI(
    rt_struct_file_path,
    dicom_file_path,
    output_dir,
    dicom_image_save_path,
    structures=None,
    gzip=True,
    mask_background_value=0,
    mask_foreground_value=255,
    convert_original_dicom=True,
    series_id=None,
):
    """
    Converts A DICOM and DICOM RT Struct file to nii

    :param rt_struct_file_path: Path to the rtstruct file
    :param dicom_file_path: Path to the dicom file
    :param output_dir: Output path where the masks are written to. Make sure trailing slash is there
    :param structures: Optional, list of structures to convert
    :param gzip: Optional, output .nii.gz if set to True, default: True
    :param series_id: Optional, the Series Instance UID. Use  to specify the ID corresponding to the image if there are
    dicoms from more than one series in `dicom_file` folder

    :raise InvalidFileFormatException: Raised when an invalid file format is given.
    :raise PathDoesNotExistException: Raised when the given path does not exist.
    :raise UnsupportedTypeException: Raised when conversion is not supported.
    :raise ValueError: Raised when mask_background_value or mask_foreground_value is invalid.
    """
    output_dir = os.path.join(output_dir, "")

    if not os.path.exists(rt_struct_file_path):
        raise PathDoesNotExistException(
            f"rtstruct path does not exist: {rt_struct_file_path}"
        )

    if not os.path.exists(dicom_file_path):
        raise PathDoesNotExistException(
            f"DICOM path does not exists: {dicom_file_path}"
        )

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

    os.makedirs(output_dir, exist_ok=True)

    rtreader = RtStructInputAdapter()

    all_rt_structs = rtreader.ingest(rt_struct_file_path)
    dicom_image = DcmInputAdapter().ingest(dicom_file_path, series_id=series_id)

    dcm_patient_coords_to_mask = DcmPatientCoords2Mask()
    nii_output_adapter = NiiOutputAdapter()
    for rtstruct in all_rt_structs:
        if len(structures) == 0 or rtstruct["name"] in structures:
            if "sequence" not in rtstruct:
                print(
                    "Skipping mask {} no shape/polygon found".format(rtstruct["name"])
                )
                continue

            print("Working on mask {}".format(rtstruct["name"]))
            try:
                mask = dcm_patient_coords_to_mask.convert(
                    rtstruct["sequence"],
                    dicom_image,
                    mask_background_value,
                    mask_foreground_value,
                )
            except ContourOutOfBoundsException:
                print(
                    f'Structure {rtstruct["name"]} is out of bounds, ignoring contour!'
                )
                continue

            mask.CopyInformation(dicom_image)

            mask_filename = f'{rtstruct["name"]}'

            # Trailing Slash present in output dir
            nii_output_adapter.write(mask, f"{output_dir}{mask_filename}", gzip)

    if convert_original_dicom:
        print("Converting original DICOM to nii")
        nii_output_adapter.write(dicom_image, dicom_image_save_path, gzip)

    print("Success!")
