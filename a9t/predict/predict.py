import os
import shutil
from datetime import datetime
from typing import List

from a9t.utils.nifti2rt import convert_nifti_outputs_to_dicom
from a9t.config import ALL_SEG_MAP, LOG, SAMPLE_NUMBER_ZFILL
from a9t.dao.table import DicomLog
from a9t.postprocess import postprocess_folder
from a9t.evaluate.evaluate import generate_labels_on_data
from a9t.preprocess.preprocess_data import convert_dicom_dir_to_nnunet_dataset
from a9t.utils.ioutils import remove_stuff, normpath


def folder_predict(dcm_logs: List[DicomLog], preds_dir, dataset_name, only_original):
    task_map = ALL_SEG_MAP[dataset_name]
    exp_number = datetime.now().strftime("%Y-%m-%d.%H-%M")
    parent_dataset_name = dataset_name

    for dataset_id in task_map.keys():
        LOG.info("Processing ID", dataset_id)
        dataset_specific_map = task_map[dataset_id]
        model_config = dataset_specific_map["config"]
        dataset_name = dataset_specific_map["name"]
        seg_map = dataset_specific_map["map"]
        trainer_name = dataset_specific_map["trainer_name"]
        postprocess = dataset_specific_map["postprocess"]

        dataset_dir = normpath(f"data/nnUNet_raw/Dataset{dataset_id}_{dataset_name}")
        remove_stuff(dataset_dir)

        dcm_input_paths = [dcm.input_path for dcm in dcm_logs]
        LOG.info(f"Found {len(dcm_input_paths)} directories to work on...")

        for idx, dicom_dir in enumerate(dcm_input_paths):
            sample_number = str(idx).zfill(SAMPLE_NUMBER_ZFILL)

            dataset_dir = convert_dicom_dir_to_nnunet_dataset(
                dicom_dir,
                dataset_id,
                dataset_name,
                sample_number,
                seg_map,
                data_tag="seg",
                extension="nii.gz",
                only_original=only_original,
            )
        tr_images = os.path.join(dataset_dir, "imagesTr")
        model_pred_dir = os.path.join(
            preds_dir, parent_dataset_name, str(dataset_id), "modelpred"
        )
        remove_stuff(model_pred_dir)

        generate_labels_on_data(
            tr_images, dataset_id, model_pred_dir, model_config, trainer_name
        )

        if postprocess is not None:
            ip_folder = model_pred_dir
            op_folder = os.path.join(
                preds_dir, parent_dataset_name, str(dataset_id), "postprocess"
            )
            remove_stuff(op_folder)
            os.makedirs(op_folder, exist_ok=True)
            pkl_file_src = postprocess
            pkl_file_dest = f"{op_folder}/postprocessing.pkl"
            # copy
            shutil.copy(pkl_file_src, pkl_file_dest)
            postprocess_folder(ip_folder, op_folder, pkl_file_dest)
            model_pred_dir = op_folder

        final_output_dir = os.path.join(preds_dir, parent_dataset_name, "results")
        convert_nifti_outputs_to_dicom(
            model_pred_dir,
            final_output_dir,
            dataset_dir,
            dataset_id,
            exp_number,
            seg_map,
        )
