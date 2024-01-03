import os
import shutil
from datetime import datetime
from functools import partial
from multiprocessing.pool import Pool
from typing import List
from draw.dao.common import Status
from draw.dao.db import DBConnection

from draw.utils import ioutils
from draw.utils.nifti2rt import convert_nifti_outputs_to_dicom
from draw.config import ALL_SEG_MAP, LOG, SAMPLE_NUMBER_ZFILL, NNUNET_RAW_DATA_ENV_KEY
from draw.dao.table import DicomLog
from draw.postprocess import postprocess_folder
from draw.evaluate.evaluate import generate_labels_on_data
from draw.preprocess.preprocess_data import convert_dicom_dir_to_nnunet_dataset
from draw.utils.ioutils import remove_stuff, normpath

POOL_WORKERS = 2


def folder_predict(dcm_logs: List[DicomLog], preds_dir, dataset_name, only_original):
    task_map = ALL_SEG_MAP[dataset_name]
    exp_number = datetime.now().strftime("%Y-%m-%d.%H-%M")
    parent_dataset_name = dataset_name

    with Pool(processes=POOL_WORKERS) as m_pool:
        # in order of keys, keys order guaranteed by python
        all_model_prediction_dirs = m_pool.map(
            partial(
                predict_one_dataset,
                dcm_logs=dcm_logs,
                only_original=only_original,
                parent_dataset_name=parent_dataset_name,
                preds_dir=preds_dir,
                task_map=task_map,
            ),
            task_map.keys(),
        )

    LOG.info(f"Prediction completed from model {dataset_name}")

    final_output_dir = get_final_output_dir(parent_dataset_name, preds_dir)

    for dataset_id, model_pred_dir, dataset_specific_map in zip(
        task_map.keys(),
        all_model_prediction_dirs,
        task_map.values(),
    ):
        ioutils.assert_env_key_set(NNUNET_RAW_DATA_ENV_KEY)
        raw_base_dir = os.getenv(NNUNET_RAW_DATA_ENV_KEY)
        dataset_specific_map = task_map[dataset_id]
        dataset_name = dataset_specific_map["name"]
        dataset_dir = normpath(f"{raw_base_dir}/Dataset{dataset_id}_{dataset_name}")
        seg_map = dataset_specific_map["map"]
        convert_nifti_outputs_to_dicom(
            model_pred_dir,
            final_output_dir,
            dataset_dir,
            dataset_id,
            exp_number,
            seg_map,
        )


def predict_one_dataset(
    dataset_id,
    dcm_logs,
    only_original,
    parent_dataset_name,
    preds_dir,
    task_map,
):
    ioutils.assert_env_key_set(NNUNET_RAW_DATA_ENV_KEY)
    raw_base_dir = os.getenv(NNUNET_RAW_DATA_ENV_KEY)
    dataset_specific_map = task_map[dataset_id]
    dataset_name = dataset_specific_map["name"]
    dataset_dir = normpath(f"{raw_base_dir}/Dataset{dataset_id}_{dataset_name}")

    LOG.info(f"Processing ID {dataset_id}")

    model_config = dataset_specific_map["config"]
    seg_map = dataset_specific_map["map"]
    trainer_name = dataset_specific_map["trainer_name"]
    postprocess = dataset_specific_map["postprocess"]

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

    conn = DBConnection()
    for dcm_log in dcm_logs:
        conn.update_status(dcm_log, Status.STARTED)
    del conn

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
    return model_pred_dir


def get_final_output_dir(parent_dataset_name, preds_dir):
    final_output_dir = os.path.join(preds_dir, parent_dataset_name, "results")
    return final_output_dir
