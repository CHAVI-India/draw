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

# Crash recovery: a study claimed by a worker that then dies sits in STARTED. The
# reaper returns such studies to the queue once their lease expires. The lease must be
# comfortably longer than a real prediction so we never re-queue work still running.
LEASE_SECONDS = 3600
MAX_ATTEMPTS = 3


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
) -> bool:
    claimed = queue.claim(seg_model_name, limit=config.pred_batch_size)
    if not claimed:
        return False

    # NB: no cooldown sleep here — sleeping after claiming would burn the lease while
    # the item sits in STARTED doing nothing, bringing it closer to a spurious reaper
    # re-queue. The inter-cycle pacing lives in task_model_prediction's GPU recheck.
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
                "config": config,
                "result_sink": sink,
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
) -> None:
    # Imported here so importing this module needs no GPU/infra at module load.
    from draw_pipeline.pipeline.gpu import get_gpu_memory

    model_names = registry.names()
    if not model_names:
        raise RuntimeError(
            "No models configured (empty registry); cannot start prediction loop. "
            "Check MODEL_DEF_ROOT points at the config_yaml dir."
        )

    required_free_mb = config.required_free_gpu_mb
    # cycle() over a non-empty list never terminates, so this is an explicit
    # forever-loop that round-robins models. (Legacy used `while next(cycle)` which
    # both obscured that AND crashed with an uncaught StopIteration on an empty list.)
    for model_name in cycle(model_names):
        try:
            # Recover studies stranded in STARTED by a previously-killed worker before
            # looking for new work, so a crash/restart self-heals.
            requeued = queue.requeue_expired(LEASE_SECONDS, MAX_ATTEMPTS)
            if requeued:
                log.warning("Reaper re-queued/failed %d stranded studies", requeued)

            gpu_memory_free = get_gpu_memory()
            any_model_ran = False
            if gpu_memory_free >= required_free_mb:
                log.info("%d MB free GPU. Trying %s", gpu_memory_free, model_name)
                any_model_ran = run_prediction(model_name, queue, registry, config)
            if not any_model_ran:
                time.sleep(GPU_RECHECK_TIME_SECONDS)
        except Exception:
            log.error("Exception ignored", exc_info=True)
            continue
