import json
import os

from draw.accessor.nnunetv2 import NNUNetV2Adapter, default_nnunet_adapter
from draw.config import (
    ALL_SEG_MAP,
    DEFAULT_FOLD,
    DATASET_JSON_FILENAME,
    PLANS_JSON_FILENAME,
    SUMMARY_JSON_FILENAME,
)

# TODO: This file is a mess. Make it no more


def get_sample_summary(s_no, summaries):
    # summaries is the array 'metric_per_case'
    s_summary = [s for s in summaries if f"seg_{s_no}" in s["reference_file"]][0]

    d = {}

    for idx in s_summary["metrics"].keys():
        ts_gyne = "TSGyne"
        d[ALL_SEG_MAP[ts_gyne][int(idx)]] = round(s_summary["metrics"][idx]["Dice"], 4)
    # metrics, 1, Dice
    return d


def evaluate_nnunet_on_folder(
    labels_dir: str,
    preds_dir: str,
    nnunet_adapter: NNUNetV2Adapter,
) -> dict:
    """
    Evaluates NNUNet on folder to generate summary.json file,
    containing scores per sample
    Args:
        nnunet_adapter: Adapter to access NNUNet
        labels_dir (str): dir containing the original labels created while preprocessing. No trailing slash
        preds_dir (str): dir containing the predictions from the model. No trailing slash

    Returns:
        dict, The ``summary.json`` file as dict containing per sample scores

    """
    dj_file = f"{preds_dir}/{DATASET_JSON_FILENAME}"
    p_file = f"{preds_dir}/{PLANS_JSON_FILENAME}"
    nnunet_adapter.evaluate_on_folder(labels_dir, preds_dir, dj_file, p_file)

    with open(f"{preds_dir}/{SUMMARY_JSON_FILENAME}", "r") as fp:
        s_file = json.load(fp)
        # As per nnunet
        return s_file["metric_per_case"]


def convert_nifti_labels_to_predictions(
    labels_dir: str,
    preds_dir: str,
):
    pass


def generate_labels_on_data(
    samples_dir,
    dataset_id,
    output_dir,
    model_config,
    trainer_name,
    nnunet_adapter: NNUNetV2Adapter = default_nnunet_adapter,
):
    os.makedirs(output_dir, exist_ok=True)
    nnunet_adapter.predict_folder(
        samples_dir, output_dir, model_config, dataset_id, DEFAULT_FOLD, trainer_name
    )


def generate_final_predicitons():
    pass
