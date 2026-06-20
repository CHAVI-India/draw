"""Prediction loop task: claim queued studies and run segmentation.

Depends only on the ``JobQueue`` Protocol (not a concrete DB), so the backing store
is swappable. ``run_prediction`` atomically ``claim``s a batch, runs ``segment_study``
(the clean core entrypoint) with a ``QueueStatusSink``, then marks the claimed items
sent. Concurrency safety comes from ``JobQueue.claim`` (atomic, SKIP LOCKED on SQL).
"""

from __future__ import annotations

import time
from itertools import cycle

from retry.api import retry_call

from draw_contracts.protocols import JobStatus
from draw_contracts.queue import JobQueue
from draw_core.config import CoreConfig
from draw_core.logging import get_logger
from draw_core.models import ModelRegistry
from draw_core.segment import segment_study
from draw_pipeline.sinks.queue_sink import QueueStatusSink

log = get_logger(__name__)

OUTPUT_DIR = "output"
PREDICTION_COOLDOWN_SECS = 30
GPU_RECHECK_TIME_SECONDS = 10


def send_to_external_server(queue: JobQueue, model_name: str) -> None:
    # TODO(draw-pipeline): wire the actual external-server upload (StorageBackend).
    for item in queue.list_by_status(model_name, JobStatus.PREDICTED, limit=1000):
        queue.mark_sent(item.series_name)
    log.info("Sent predicted results for %s", model_name)


def run_prediction(
    seg_model_name: str,
    queue: JobQueue,
    registry: ModelRegistry,
    config: CoreConfig,
    adapter,
) -> bool:
    claimed = queue.claim(seg_model_name, limit=config.pred_batch_size)
    if not claimed:
        return False

    time.sleep(PREDICTION_COOLDOWN_SECS)
    model = registry.get(seg_model_name)
    sink = QueueStatusSink(queue)
    dicom_dirs = [item.input_path for item in claimed]

    try:
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
    except Exception:
        # Prediction exhausted retries: mark the claimed items failed so they don't
        # sit forever in STARTED (and surface in the audit trail).
        log.error("Prediction failed for %s after retries", seg_model_name, exc_info=True)
        for item in claimed:
            queue.mark_failed(item.series_name, "prediction failed after retries")
        return True

    send_to_external_server(queue, seg_model_name)
    return True


def task_model_prediction(
    queue: JobQueue,
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
                any_model_ran = run_prediction(model_name, queue, registry, config, adapter)
            if not any_model_ran:
                time.sleep(GPU_RECHECK_TIME_SECONDS)
        except Exception:
            log.error("Exception ignored", exc_info=True)
            continue
