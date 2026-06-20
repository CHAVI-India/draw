"""High-level segmentation policy: DICOM study -> RT-Struct outputs.

Ported from ``draw/predict/predict.py``'s ``folder_predict``, recast as a clean
``segment_study`` that takes its dependencies explicitly (Dependency Inversion):

* a parsed ``ModelConfig`` instead of the ``ALL_SEG_MAP`` global,
* an ``NNUNetV2Adapter`` and ``CoreConfig`` rather than env reads,
* a ``StatusSink`` that decides where per-series progress is recorded — the core no
  longer reaches into the DAO. It returns a ``SegmentationResult`` for the caller.

The legacy code ran the submodels in a ``multiprocessing.Pool``. That required the
adapter/config to be picklable and is being superseded by the in-process warm
predictor (GPU perf levers 1/2). We run submodels sequentially here for correctness
and simplicity; the warm-pool path is the future (see ``accessor/warm_predictor``).
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime

from draw_contracts.dto import SegmentationResult, SeriesResult
from draw_conversion.nifti2rt import convert_nifti_outputs_to_dicom
from draw_core.accessor.predictor import Predictor, SubprocessPredictor
from draw_core.config import CoreConfig
from draw_core.constants import SAMPLE_NUMBER_ZFILL
from draw_core.logging import get_logger
from draw_core.models import ModelConfig, SubModel
from draw_core.postprocess.postprocess import postprocess_folder
from draw_core.preprocess.preprocess_data import convert_dicom_dir_to_nnunet_dataset

_default_log = get_logger(__name__)


def _remove(path: str, log: logging.Logger) -> None:
    if os.path.exists(path):
        log.info("Deleting %s", path)
        shutil.rmtree(path)


def _build_predictor(adapter, config: CoreConfig, log: logging.Logger) -> Predictor:
    """Pick the inference backend from config. Warm import is lazy (keeps core
    torch-free unless the warm path is actually requested)."""
    if config.use_warm_predictor:
        from draw_core.accessor.warm_predictor import WarmPredictor

        log.info("Using warm in-process predictor (GPU levers 1/2/5/8)")
        return WarmPredictor(config, gpu_id=config.gpu_id)
    return SubprocessPredictor(adapter)


def _predict_one_submodel(
    submodel: SubModel,
    dicom_dirs: list[str],
    preds_dir: str,
    parent_model_name: str,
    only_original: bool,
    predictor: Predictor,
    adapter,
    config: CoreConfig,
    log: logging.Logger,
) -> str:
    """Preprocess all studies for one submodel, run inference, optionally postprocess.

    Returns the directory holding this submodel's predictions.
    """
    dataset_id = submodel.dataset_id
    dataset_dir = os.path.normpath(
        f"{config.nnunet_raw_dir}/Dataset{dataset_id}_{submodel.name}"
    )
    log.info("Processing ID %s", dataset_id)
    _remove(dataset_dir, log)

    log.info("Found %d DICOM directories", len(dicom_dirs))
    for idx, dicom_dir in enumerate(dicom_dirs):
        sample_number = str(idx).zfill(SAMPLE_NUMBER_ZFILL)
        dataset_dir = convert_dicom_dir_to_nnunet_dataset(
            dicom_dir,
            dataset_id,
            submodel.name,
            sample_number,
            submodel.seg_map,
            raw_dir=config.nnunet_raw_dir,
            only_original=only_original,
            logger=log,
        )

    tr_images = os.path.join(dataset_dir, "imagesTr")
    model_pred_dir = os.path.join(preds_dir, parent_model_name, str(dataset_id), "modelpred")
    _remove(model_pred_dir, log)
    os.makedirs(model_pred_dir, exist_ok=True)
    predictor.predict_submodel(submodel, tr_images, model_pred_dir)

    if submodel.postprocess is not None:
        op_folder = os.path.join(preds_dir, parent_model_name, str(dataset_id), "postprocess")
        _remove(op_folder, log)
        os.makedirs(op_folder, exist_ok=True)
        pkl_file_dest = f"{op_folder}/postprocessing.pkl"
        shutil.copy(submodel.postprocess, pkl_file_dest)
        postprocess_folder(model_pred_dir, op_folder, pkl_file_dest, adapter)
        model_pred_dir = op_folder

    return model_pred_dir


def segment_study(
    dicom_dirs: list[str],
    preds_dir: str,
    model: ModelConfig,
    adapter,
    config: CoreConfig,
    result_sink,
    logger: logging.Logger | None = None,
    only_original: bool = True,
    predictor: Predictor | None = None,
) -> SegmentationResult:
    """Segment a set of DICOM study directories with ``model`` and record results.

    For each submodel: convert each DICOM dir to an nnU-Net dataset, run inference,
    optionally postprocess, then convert predicted NIfTIs to RT-Structs. Each
    produced series is reported to ``result_sink``. Returns the aggregate result.

    A single ``predictor`` is built once and reused across all submodels — with the
    warm predictor that collapses repeated model loads into one (GPU perf lever 2).
    If not injected, it is chosen from ``config.use_warm_predictor``.
    """
    log = logger or _default_log
    exp_number = datetime.now().strftime("%Y-%m-%d.%H-%M")
    final_output_dir = os.path.join(preds_dir, model.name, "results")
    all_series: list[SeriesResult] = []

    if predictor is None:
        predictor = _build_predictor(adapter, config, log)

    for submodel in model.submodels.values():
        model_pred_dir = _predict_one_submodel(
            submodel, dicom_dirs, preds_dir, model.name, only_original,
            predictor, adapter, config, log,
        )
        dataset_dir = os.path.normpath(
            f"{config.nnunet_raw_dir}/Dataset{submodel.dataset_id}_{submodel.name}"
        )
        series = convert_nifti_outputs_to_dicom(
            model_pred_dir,
            final_output_dir,
            dataset_dir,
            submodel.dataset_id,
            exp_number,
            submodel.seg_map,
        )
        for sr in series:
            result_sink.record_predicted(sr.series_name, sr.output_path)
            all_series.append(sr)

    log.info("Prediction complete for %s", model.name)
    return SegmentationResult(
        job_id=exp_number,
        output_local_dir=final_output_dir,
        series=all_series,
    )
