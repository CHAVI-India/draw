import os
import os.path

import SimpleITK as sitk
import numpy as np
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

from draw.config import DEFAULT_MASK_NAME
from draw.config import LOG


class DicomConverters:
    """
    Functions for Handling DICOM files
    """

    @staticmethod
    def convert_DICOM_to_Multi_NIFTI(
        rt_struct_file_path,
        dicom_file_path,
        output_dir,
        dicom_image_save_path,
        structures=None,
        gzip=True,
        mask_background_value=0,
        mask_foreground_value=255,
        series_id=None,
        only_original=True,
        save_default_empty=True,
    ):
        """Converts a DICOM and DICOM RT Struct file to NIfTI format.

        Args:
            save_default_empty:
            only_original:
            mask_foreground_value:
            mask_background_value:
            dicom_image_save_path:
            rt_struct_file_path (str): Path to the rtstruct file.
            dicom_file_path (str): Path to the dicom file.
            output_dir (str): Output path where the masks are written to. Make sure trailing slash is there.
            structures (list, optional): List of structures to convert.
            gzip (bool, optional): Output .nii.gz if set to True, default: True.
            series_id (str, optional): The Series Instance UID. Use to specify the ID corresponding to the
                image if there are dicoms from more than one series in `dicom_file` folder.

        Raises:
            InvalidFileFormatException: Raised when an invalid file format is given.
            PathDoesNotExistException: Raised when the given path does not exist.
            UnsupportedTypeException: Raised when conversion is not supported.
            ValueError: Raised when mask_background_value or mask_foreground_value is invalid.
        """

        output_dir = os.path.join(output_dir, "")

        if not os.path.exists(dicom_file_path):
            raise PathDoesNotExistException(
                f"DICOM path does not exists: {dicom_file_path}"
            )

        dicom_image = DcmInputAdapter().ingest(dicom_file_path, series_id=series_id)
        nii_output_adapter = NiiOutputAdapter()

        if not only_original:
            if not os.path.exists(rt_struct_file_path):
                raise PathDoesNotExistException(
                    f"rtstruct path does not exist: {rt_struct_file_path}"
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

            dcm_patient_coords_to_mask = DcmPatientCoords2Mask()
            for rtstruct in all_rt_structs:
                if len(structures) == 0 or rtstruct["name"] in structures:
                    if "sequence" not in rtstruct:
                        LOG.info(
                            "Skipping mask {} no shape/polygon found".format(
                                rtstruct["name"]
                            )
                        )
                        continue

                    LOG.info("Working on mask {}".format(rtstruct["name"]))
                    try:
                        mask = dcm_patient_coords_to_mask.convert(
                            rtstruct["sequence"],
                            dicom_image,
                            mask_background_value,
                            mask_foreground_value,
                        )
                    except ContourOutOfBoundsException:
                        LOG.info(
                            f'Structure {rtstruct["name"]} is out of bounds, ignoring contour!'
                        )
                        continue

                    mask.CopyInformation(dicom_image)

                    mask_filename = f'{rtstruct["name"]}'
                    LOG.info(f"Saving Mask at: {mask_filename}")

                    # Trailing Slash present in output dir
                    nii_output_adapter.write(mask, f"{output_dir}{mask_filename}", gzip)

        LOG.info(f"Converting DICOM scan to NII for path {dicom_file_path}")
        nii_output_adapter.write(dicom_image, dicom_image_save_path, gzip)

        if save_default_empty:
            nda = sitk.GetArrayFromImage(dicom_image)
            empty_mask = sitk.GetImageFromArray(np.zeros_like(nda))
            empty_mask.CopyInformation(dicom_image)
            default_empty_mask_path = f"{output_dir}{DEFAULT_MASK_NAME}"
            nii_output_adapter.write(empty_mask, default_empty_mask_path, gzip)
            LOG.info(f"Saved Default Empty Mask at {default_empty_mask_path}")

        LOG.info(f"Conversion for {dicom_file_path} complete")
        LOG.info(f"Final output saved at {dicom_image_save_path}")
