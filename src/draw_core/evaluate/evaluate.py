"""Run nnU-Net inference and evaluation on a folder.

Ported from ``draw/evaluate/evaluate.py``. Changes:

* The adapter is a required argument (no import-time ``default_nnunet_adapter``).
* The dead ``convert_nifti_labels_to_predictions`` and ``generate_final_predicitons``
  pass-stubs are removed.
* The broken ``get_sample_summary`` (it hardcoded the ``"TSGyne"`` model and read the
  removed ``ALL_SEG_MAP``) now takes the ``seg_map`` as a parameter.
"""

from __future__ import annotations

import json
import os

from draw_core.accessor.nnunetv2 import NNUNetV2Adapter
from draw_core.constants import (
    DATASET_JSON_FILENAME,
    DEFAULT_FOLD,
    PLANS_JSON_FILENAME,
    SAMPLE_SEP_DELIM,
    SUMMARY_JSON_FILENAME,
)


def generate_labels_on_data(
    samples_dir: str,
    dataset_id: str,
    output_dir: str,
    model_config: str,
    trainer_name: str,
    adapter: NNUNetV2Adapter,
    fold: str = DEFAULT_FOLD,
) -> None:
    """Run nnU-Net inference on ``samples_dir`` into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    adapter.predict_folder(
        samples_dir, output_dir, model_config, str(dataset_id), fold, trainer_name
    )


def evaluate_nnunet_on_folder(
    labels_dir: str,
    preds_dir: str,
    adapter: NNUNetV2Adapter,
) -> list:
    """Evaluate predictions against ground-truth labels; return per-case metrics.

    Args:
        labels_dir: dir with the original labels created during preprocessing.
        preds_dir: dir with the model predictions (also holds dataset/plans json).
        adapter: nnU-Net adapter.

    Returns:
        The ``metric_per_case`` array from nnU-Net's ``summary.json``.
    """
    dj_file = f"{preds_dir}/{DATASET_JSON_FILENAME}"
    p_file = f"{preds_dir}/{PLANS_JSON_FILENAME}"
    adapter.evaluate_on_folder(labels_dir, preds_dir, dj_file, p_file)

    with open(f"{preds_dir}/{SUMMARY_JSON_FILENAME}") as fp:
        return json.load(fp)["metric_per_case"]


def get_sample_summary(
    sample_no: str,
    summaries: list,
    seg_map: dict[int, str],
) -> dict[str, float]:
    """Per-class Dice scores for one sample, keyed by structure name.

    ``seg_map`` is passed in explicitly (the legacy version hardcoded ``"TSGyne"``).
    """
    s_summary = next(
        s for s in summaries if f"{SAMPLE_SEP_DELIM}{sample_no}" in s["reference_file"]
    )
    return {
        seg_map[int(idx)]: round(metrics["Dice"], 4)
        for idx, metrics in s_summary["metrics"].items()
    }
