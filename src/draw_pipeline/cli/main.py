"""DRAW CLI entrypoint.

Ports ``draw/main.py`` + ``draw/cli/*`` into a single click group. Exposes the SAME
five commands and flags as before: ``train-single-gpu``, ``predict``, ``preprocess``,
``start-pipeline``, ``zip-model``.

Each command configures logging first, loads the ``RuntimeEnv`` + ``ModelRegistry``,
builds a ``CoreConfig`` + ``NNUNetV2Adapter``, and calls into the core. The model
registry is loaded inside the command (not at import) so importing this module needs
no env file. Model-name choices are validated against ``ModelRegistry.names()``.
"""

from __future__ import annotations

import multiprocessing

import click

from draw_core.config import CoreConfig
from draw_core.constants import MODEL_FOLDS
from draw_core.logging import configure_logging, get_logger
from draw_core.models import ModelRegistry
from draw_pipeline.config import load_env

log = get_logger(__name__)


def _setup_logging() -> None:
    """Configure logging from env vars so deployments control it without code changes.

    DRAW_LOG_FILE  - if set, also write rotating daily logs there (30-day retention,
                     gzip-compressed rotations). Point it at a mounted volume in Docker.
    DRAW_LOG_LEVEL - log level (default INFO).
    """
    import os

    configure_logging(
        level=os.environ.get("DRAW_LOG_LEVEL", "INFO"),
        logfile=os.environ.get("DRAW_LOG_FILE"),
    )


def _bootstrap() -> tuple[object, ModelRegistry]:
    """Configure logging and load the env + model registry. Entrypoint helper."""
    _setup_logging()
    env = load_env()
    registry = ModelRegistry.from_yaml_dir(env.model_def_root)
    return env, registry


def _resolve_model(registry: ModelRegistry, name: str):
    if name not in registry.names():
        raise click.BadParameter(
            f"{name!r} is not a known model. Choose from: {registry.names()}"
        )
    return registry.get(name)


@click.group(name="draw", help="AutoSegmentation Pipeline based on NNUNet")
def cli():
    pass


@cli.command("train-single-gpu", help="Prepare and Train model on a single GPU")
@click.option("--model-fold", type=click.Choice(MODEL_FOLDS), default="0", help="Fold of data to Train")
@click.option("--gpu-id", type=int, default=0, help="GPU id.")
@click.option("--model-name", type=str, required=True, help="Name of Model")
@click.option("--dataset-id", type=int, required=True, help="3 digit dataset ID")
@click.option("--gpu-space", type=int, default=None, help="GPU space in GB. [WARN] Use carefully")
@click.option("--email-address", type=str, default=None, help="Email address to send notification")
@click.option("--determine-postprocessing", is_flag=True, help="Enable or disable postprocessing determination")
@click.option("--train-continue", is_flag=True, help="Resume training from where left off")
def cli_prepare_and_train(
    model_name, model_fold, gpu_id, dataset_id, gpu_space,
    email_address, determine_postprocessing, train_continue,
):
    from draw_core.accessor.nnunetv2 import NNUNetV2Adapter
    from draw_core.engines.base import TrainableEngine
    from draw_core.engines.factory import build_engine
    from draw_pipeline.train import prepare_and_train

    _env, registry = _bootstrap()
    model = _resolve_model(registry, model_name)
    config = CoreConfig()

    # Interface segregation: only engines that declare the training capability can be
    # trained. A remote/ONNX/import-only engine fails here with a clear message
    # instead of pretending to train.
    engine = build_engine(model.engine, model.engine_config, core_config=config)
    if not isinstance(engine, TrainableEngine):
        raise click.ClickException(
            f"Engine {model.engine!r} for model {model_name!r} does not support training."
        )

    adapter = NNUNetV2Adapter(config)
    log.warning("Make sure you ran preprocess before this. Ignore if you did.")
    prepare_and_train(
        model, model_fold, gpu_id, dataset_id, gpu_space, email_address,
        determine_postprocessing, train_continue, adapter, config,
    )


@cli.command("predict", help="Generate Predictions from trained model")
@click.option("--preds-dir", "-p", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True), required=True, help="Output Directory that will contain final labels")
@click.option("--root-dir", "-r", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True), required=True, help="Directory containing other DICOM parent directories")
@click.option("--dataset-name", "-n", type=str, required=True, help="Name of the dataset")
@click.option("--only-original", is_flag=True, help="Convert only original DICOM. Set this to disable RTStruct file searching and parsing")
@click.option("--warm", is_flag=True, help="Use the in-process resident predictor (GPU perf: load weights once, reuse across submodels). Requires the 'gpu' extra.")
@click.option("--gpu-id", type=int, default=None, help="Pin inference to a specific GPU / MIG slice (sets CUDA_VISIBLE_DEVICES).")
def cli_predict(preds_dir, dataset_name, root_dir, only_original, warm, gpu_id):
    import os

    from draw_contracts.sink import NullStatusSink
    from draw_core.segment import segment_study

    _env, registry = _bootstrap()
    model = _resolve_model(registry, dataset_name)
    # --warm/--gpu-id are nnU-Net runtime knobs; they flow to the engine via CoreConfig.
    config = CoreConfig(use_warm_predictor=warm, gpu_id=gpu_id)
    dicom_dirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    segment_study(
        dicom_dirs=dicom_dirs,
        preds_dir=preds_dir,
        model=model,
        config=config,
        result_sink=NullStatusSink(),
    )


@cli.command("preprocess", help="Preprocess DICOM Data to nnUNet format")
@click.option("--root-dir", "-d", type=str, required=True, help="Parent Directory containing other DICOM directories")
@click.option("--dataset-id", "-i", type=str, required=True, help="3 digit ID of the dataset")
@click.option("--dataset-name", "-n", type=str, required=True, help="Name of the dataset from given list")
@click.option("--start", "-s", type=int, required=False, default=0, help="The sample number to start putting data from")
@click.option("--only-original", is_flag=True, help="Convert only original DICOM. Set this to disable RTStruct file searching and parsing")
def cli_preprocess(root_dir, dataset_id, dataset_name, start, only_original):
    from draw_core.preprocess.preprocess_data import run_pre_processing

    _env, registry = _bootstrap()
    model = _resolve_model(registry, dataset_name)
    config = CoreConfig()
    run_pre_processing(
        int(dataset_id), model, only_original, root_dir, start, raw_dir=config.nnunet_raw_dir
    )


@cli.command("start-pipeline", help="Starts Continuous Prediction Pipeline")
def cli_start_pipeline():
    from draw_pipeline.pipeline.start import start_continuous_prediction

    multiprocessing.freeze_support()
    _setup_logging()
    env = load_env()
    config = CoreConfig()
    start_continuous_prediction(env, config)


@cli.group("db", help="Database schema management (Alembic migrations)")
def db_group():
    pass


@db_group.command("upgrade", help="Apply migrations up to a revision (default: head)")
@click.argument("revision", default="head")
def cli_db_upgrade(revision):
    from draw_pipeline import migrations

    _setup_logging()
    migrations.upgrade(load_env().db_url, revision)


@db_group.command("downgrade", help="Revert to an earlier revision")
@click.argument("revision")
def cli_db_downgrade(revision):
    from draw_pipeline import migrations

    _setup_logging()
    migrations.downgrade(load_env().db_url, revision)


@db_group.command("current", help="Show the current schema revision")
def cli_db_current():
    from draw_pipeline import migrations

    _setup_logging()
    migrations.current(load_env().db_url)


@cli.command("zip-model", help="Convert model into ZIP archive")
@click.option("--dataset-id", type=int, required=True, help="3-digit ID for the dataset")
@click.option("--model-name", type=str, required=True, help="Model name")
def cli_export(dataset_id, model_name):
    from draw_pipeline.impex.export import export_to_zip

    _env, registry = _bootstrap()
    model = _resolve_model(registry, model_name)
    export_to_zip(dataset_id, model, CoreConfig())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    cli()
