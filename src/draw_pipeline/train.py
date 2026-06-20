"""Single-GPU prepare + train flow.

Ported from ``draw/train/train.py``. The adapter is injected (legacy used the global
``default_nnunet_adapter``) and the submodel comes from a parsed ``ModelConfig``.
"""

from __future__ import annotations

import glob
import os
import shutil

from draw_core.accessor.nnunetv2 import NNUNetV2Adapter
from draw_core.config import CoreConfig
from draw_core.constants import DATASET_JSON_FILENAME, PLANS_JSON_FILENAME
from draw_core.logging import get_logger
from draw_core.models import ModelConfig

log = get_logger(__name__)


def copy_files(files_to_copy: list[str], results_dir: str) -> None:
    log.info("Copying %s to %s", files_to_copy, results_dir)
    os.makedirs(results_dir, exist_ok=True)
    for file_path in files_to_copy:
        destination_path = os.path.join(results_dir, os.path.basename(file_path))
        try:
            shutil.copy(file_path, destination_path)
            log.debug("Copied '%s' to '%s'", os.path.basename(file_path), results_dir)
        except Exception as e:
            log.debug("Error copying '%s': %s", file_path, e)


def prepare_and_train(
    model: ModelConfig,
    model_fold: str,
    gpu_id: int,
    dataset_id: int,
    gpu_space: int | None,
    email_address: str | None,
    determine_postprocessing: bool,
    train_continue: bool,
    adapter: NNUNetV2Adapter,
    config: CoreConfig,
) -> None:
    submodel = model.submodels[int(dataset_id)]
    trainer_name = submodel.trainer_name
    model_config = submodel.config
    model_name = submodel.name

    log.info("Starting planning for %s", dataset_id)
    adapter.plan(str(dataset_id), config=model_config, gpu_memory_gb=gpu_space)

    log.info(
        "Starting training for %s, fold %s, trainer %s, device=%s",
        dataset_id, model_fold, trainer_name, gpu_id,
    )
    adapter.train(
        str(dataset_id), model_config, model_fold, trainer_name,
        resume=train_continue, device_id=gpu_id,
    )
    log.info("Completed training for %s", dataset_id)

    if determine_postprocessing:
        dj_file, gt_dir, p_file, preds_dir, results_dir = get_evaluation_file_paths(
            dataset_id, model_config, model_name, trainer_name, model_fold, config
        )
        adapter.evaluate_on_folder(
            gt_dir=gt_dir, preds_dir=preds_dir, dj_file=dj_file, p_file=p_file
        )
        log.info("Evaluation complete for %s", dataset_id)
        adapter.determine_postprocessing(
            input_folder=preds_dir, gt_labels_folder=gt_dir, dj_file=dj_file, p_file=p_file
        )
        copy_files(glob.glob(f"{preds_dir}/postprocessing**"), results_dir)
        log.info("Postprocessing determined for %s", dataset_id)

    if email_address is not None:
        log.info("Sending email to %s", email_address)


def get_evaluation_file_paths(
    dataset_id: int,
    model_config: str,
    model_name: str,
    trainer_name: str,
    fold_no: str,
    config: CoreConfig,
) -> tuple[str, str, str, str, str]:
    results_dir = os.path.normpath(
        f"{config.nnunet_results_dir}/Dataset{dataset_id}_{model_name}"
        f"/{trainer_name}__nnUNetPlans__{model_config}"
    )
    gt_dir = os.path.normpath(
        f"{config.nnunet_preprocessed_dir}/Dataset{dataset_id}_{model_name}/gt_segmentations"
    )
    preds_dir = os.path.normpath(f"{results_dir}/fold_{fold_no}/validation")
    os.makedirs(preds_dir, exist_ok=True)
    dj_file = os.path.normpath(f"{results_dir}/{DATASET_JSON_FILENAME}")
    p_file = os.path.normpath(f"{results_dir}/{PLANS_JSON_FILENAME}")
    return dj_file, gt_dir, p_file, preds_dir, results_dir
