import glob
import os

from a9t.accessor.nnunetv2 import default_nnunet_adapter
from a9t.config import (
    LOG,
    ALL_SEG_MAP,
    NNUNET_RAW_DATA_ENV_KEY,
    NNUNET_RESULTS_DATA_ENV_KEY,
    DATASET_JSON_FILENAME,
    PLANS_JSON_FILENAME,
)
from a9t.utils.ioutils import normpath


def copy_files(postprocessing_files, results_dir):
    LOG.info(f"Copying {postprocessing_files} to {results_dir}")
    pass


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


def get_evaluation_file_paths(dataset_id, model_config, model_name, trainer_name):
    # For whole dataset, if you do for only validation, then change paths
    results_dir = normpath(
        f"{os.environ[NNUNET_RESULTS_DATA_ENV_KEY]}"
        f"/Dataset{dataset_id}_{model_name}"
        f"/{trainer_name}__nnUNetPlans__{model_config}"
    )

    gt_dir = normpath(
        f"{os.environ[NNUNET_RAW_DATA_ENV_KEY]}"
        f"/Dataset{dataset_id}_{model_name}"
        f"/labelsTr"
    )
    preds_dir = normpath(
        f"{os.environ[NNUNET_RAW_DATA_ENV_KEY]}"
        f"/Dataset{dataset_id}_{model_name}"
        f"/predsTr"
    )
    dj_file = normpath(f"{results_dir}/{DATASET_JSON_FILENAME}")
    p_file = normpath(f"{results_dir}/{PLANS_JSON_FILENAME}")
    return dj_file, gt_dir, p_file, preds_dir, results_dir
