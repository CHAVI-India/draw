import glob
import os
import shutil

from a9t.accessor.nnunetv2 import default_nnunet_adapter
from a9t.config import (
    LOG,
    ALL_SEG_MAP,
    NNUNET_PREPROCESSED_ENV_KEY,
    NNUNET_RAW_DATA_ENV_KEY,
    NNUNET_RESULTS_DATA_ENV_KEY,
    DATASET_JSON_FILENAME,
    PLANS_JSON_FILENAME,
)
from a9t.utils.ioutils import normpath


def copy_files(files_to_copy, results_dir):
    LOG.info(f"Copying {files_to_copy} to {results_dir}")
    # Ensure the destination directory exists, create it if necessary
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Copy each file to the destination directory
    for file_path in files_to_copy:
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(results_dir, file_name)

        try:
            shutil.copy(file_path, destination_path)
            LOG.debug(f"Successfully copied '{file_name}' to '{results_dir}'")
        except Exception as e:
            LOG.debug(f"Error copying '{file_name}': {e}")


def prepare_and_train(
    model_name: str,
    model_fold: str,
    gpu_id: int,
    dataset_id: int,
    gpu_space: int,
    email_address,
    determine_postprocessing: bool,
    train_continue: bool,
):
    dataset_map = ALL_SEG_MAP[model_name][dataset_id]
    trainer_name = dataset_map["trainer_name"]
    model_config = dataset_map["config"]
    model_name = dataset_map["name"]

    LOG.info(f"Starting Planning for {dataset_id}")
    default_nnunet_adapter.plan(
        dataset_id, config=model_config, gpu_memory_gb=gpu_space
    )

    LOG.info(
        f"Starting Training for {dataset_id}, fold {model_fold}, Trainer: {trainer_name}, device={gpu_id}"
    )
    default_nnunet_adapter.train(
        dataset_id,
        model_config,
        model_fold,
        trainer_name,
        resume=train_continue,
        device_id=gpu_id,
    )

    LOG.info(f"Completed Training for {dataset_id}")

    if determine_postprocessing:
        dj_file, gt_dir, p_file, preds_dir, results_dir = get_evaluation_file_paths(
            dataset_id,
            model_config,
            model_name,
            trainer_name,
            model_fold,
        )

        default_nnunet_adapter.evaluate_on_folder(
            gt_dir=gt_dir,
            preds_dir=preds_dir,
            dj_file=dj_file,
            p_file=p_file,
        )

        LOG.info(f"Evaluation Complete for {dataset_id}")

        default_nnunet_adapter.determine_postprocessing(
            input_folder=preds_dir,
            gt_labels_folder=gt_dir,
            dj_file=dj_file,
            p_file=p_file,
        )
        # Input folder has postprocessing files
        postprocessing_files = glob.glob(f"{preds_dir}/postprocessing**")
        copy_files(postprocessing_files, results_dir)
        LOG.info(f"PostProcessing Determined for {dataset_id}")

    if email_address is not None:
        LOG.info(f"Sending Email to {email_address}")


def get_evaluation_file_paths(
    dataset_id, model_config, model_name, trainer_name, fold_no
):
    # Only for Validation, There will be warnings
    # Val Pred Dir: data/nnUNet_results/Dataset722_TSPrimeCTVN/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation
    # Val GT Dir: data/nnUNet_preprocessed/Dataset722_TSPrimeCTVN/gt_segmentations
    results_dir = normpath(
        f"{os.environ[NNUNET_RESULTS_DATA_ENV_KEY]}"
        f"/Dataset{dataset_id}_{model_name}"
        f"/{trainer_name}__nnUNetPlans__{model_config}"
    )

    gt_dir = normpath(
        f"{os.environ[NNUNET_PREPROCESSED_ENV_KEY]}"
        f"/Dataset{dataset_id}_{model_name}"
        f"/gt_segmentations"
    )

    preds_dir = normpath(f"{results_dir}/fold_{0}/validation")
    os.makedirs(preds_dir, exist_ok=True)

    dj_file = normpath(f"{results_dir}/{DATASET_JSON_FILENAME}")
    p_file = normpath(f"{results_dir}/{PLANS_JSON_FILENAME}")

    return dj_file, gt_dir, p_file, preds_dir, results_dir
