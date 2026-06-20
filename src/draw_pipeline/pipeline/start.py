"""Start the continuous prediction pipeline: watcher + prediction loop processes.

Ported from ``draw/pipeline/start.py``. Builds dependencies (engine, DB connection,
model registry, core config, adapter) and hands them to the two long-running tasks.
Each task runs in its own process and is given its own dependency objects.
"""

from __future__ import annotations

from multiprocessing import Process

from draw_core.config import CoreConfig
from draw_core.logging import get_logger
from draw_core.models import ModelRegistry
from draw_pipeline.config import RuntimeEnv
from draw_pipeline.dao.common import ensure_schema, get_engine
from draw_pipeline.dao.db import DBConnection

log = get_logger(__name__)


def _run_watcher(env: RuntimeEnv) -> None:
    from draw_pipeline.pipeline.task_copy import task_watch_dir

    db = DBConnection(get_engine(env.db_url))
    registry = ModelRegistry.from_yaml_dir(env.model_def_root)
    task_watch_dir(env.watch_dir, db, registry)


def _run_predictor(env: RuntimeEnv, config: CoreConfig) -> None:
    from draw_core.accessor.nnunetv2 import NNUNetV2Adapter
    from draw_pipeline.pipeline.task_predict import task_model_prediction

    db = DBConnection(get_engine(env.db_url), batch_size=config.pred_batch_size)
    registry = ModelRegistry.from_yaml_dir(env.model_def_root)
    adapter = NNUNetV2Adapter(config)
    task_model_prediction(db, registry, config, adapter)


def start_continuous_prediction(env: RuntimeEnv, config: CoreConfig) -> None:
    # Startup prerequisite for the long-lived pipeline: make sure the queue table
    # exists. Idempotent and cheap, so a fresh deployment "just works" without a
    # manual `alembic upgrade head`. One-shot CLI commands (predict/preprocess/
    # zip-model) deliberately do NOT call this — they don't touch the queue table.
    log.info("Ensuring DB schema exists at %s", env.db_url)
    ensure_schema(get_engine(env.db_url))

    processes = [
        Process(target=_run_predictor, args=(env, config), name="predictor"),
        Process(target=_run_watcher, args=(env,), name="watcher"),
    ]
    for p in processes:
        log.info("Starting %s", p.name)
        p.start()
    for p in processes:
        p.join()
