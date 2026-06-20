"""Prediction loop task: claim queued studies and run segmentation.

Ported from ``draw/pipeline/TASK_predict.py``. ``run_prediction`` now builds a
``SqlStatusSink`` + ``NNUNetV2Adapter`` and calls ``segment_study`` (the clean core
entrypoint) instead of the old ``folder_predict``. The DB connection, model registry
and config are injected.
"""

from __future__ import annotations

import time
from itertools import cycle

from retry.api import retry_call

from draw_core.config import CoreConfig
from draw_core.logging import get_logger
from draw_core.models import ModelRegistry
from draw_core.segment import segment_study
from draw_pipeline.dao.common import Status
from draw_pipeline.dao.db import DBConnection
from draw_pipeline.sinks.sql_sink import SqlStatusSink

log = get_logger(__name__)

OUTPUT_DIR = "output"
PREDICTION_COOLDOWN_SECS = 30
GPU_RECHECK_TIME_SECONDS = 10


def send_to_external_server(db: DBConnection, model_name: str) -> None:
    # TODO(draw-pipeline): wire the actual external-server upload (StorageBackend).
    pred = db.top(model_name, Status.PREDICTED)
    for dcm in pred:
        db.update_status_by_id(dcm, Status.SENT)
    log.info("Sent %d to server", len(pred))


def run_prediction(
    seg_model_name: str,
    db: DBConnection,
    registry: ModelRegistry,
    config: CoreConfig,
    adapter,
) -> bool:
    all_dcm_files = db.dequeue(seg_model_name)
    if not all_dcm_files:
        return False

    time.sleep(PREDICTION_COOLDOWN_SECS)
    model = registry.get(seg_model_name)
    sink = SqlStatusSink(db)
    dicom_dirs = [dcm.input_path for dcm in all_dcm_files]

    retry_call(
        segment_study,
        fkwargs={
            "dicom_dirs": dicom_dirs,
            "preds_dir": OUTPUT_DIR,
            "model": model,
            "adapter": adapter,
            "config": config,
            "result_sink": sink,
            "only_original": True,
        },
        tries=2,
        logger=log,
        delay=PREDICTION_COOLDOWN_SECS,
    )

    send_to_external_server(db, seg_model_name)
    return True


def task_model_prediction(
    db: DBConnection,
    registry: ModelRegistry,
    config: CoreConfig,
    adapter,
) -> None:
    # Imported here so importing this module needs no GPU/infra at module load.
    from draw_pipeline.pipeline.gpu import get_gpu_memory

    required_free_mb = config.required_free_gpu_mb
    model_name_generator = cycle(registry.names())
    while model_name := next(model_name_generator):
        try:
            gpu_memory_free = get_gpu_memory()
            any_model_ran = False
            if gpu_memory_free >= required_free_mb:
                log.info("%d MB free GPU. Trying %s", gpu_memory_free, model_name)
                any_model_ran = run_prediction(model_name, db, registry, config, adapter)
            if not any_model_ran:
                log.info("%s ran: %s", model_name, any_model_ran)
                time.sleep(GPU_RECHECK_TIME_SECONDS)
        except Exception:
            log.error("Exception ignored", exc_info=True)
            continue
