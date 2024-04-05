import click

from draw.config import MODEL_FOLDS, ALL_SEG_MAP, LOG
from draw.train.train import prepare_and_train


@click.command(help="Prepare and Train model on a single GPU")
@click.option(
    "--model-fold",
    type=click.Choice(MODEL_FOLDS),
    default="0",
    help="Fold of data to Train",
)
@click.option(
    "--gpu-id",
    type=int,
    default=0,
    help="GPU id.",
)
@click.option(
    "--model-name",
    type=click.Choice(ALL_SEG_MAP.keys()),
    help="Name of Model",
    required=True,
)
@click.option(
    "--dataset-id",
    type=int,
    help="3 digit dataset ID",
    required=True,
)
@click.option(
    "--gpu-space",
    type=int,
    help="GPU space in GB. [WARN] Use carefully",
    default=None,
)
@click.option(
    "--email-address",
    type=str,
    help="Email address to send notification",
    default=None,
)
@click.option(
    "--determine-postprocessing",
    is_flag=True,
    help="Enable or disable postprocessing determination",
)
@click.option(
    "--train-continue",
    is_flag=True,
    help="Resume training from where left off",
)
def cli_prepare_and_train(
    model_name,
    model_fold,
    gpu_id,
    dataset_id,
    gpu_space,
    email_address,
    determine_postprocessing,
    train_continue,
):
    """
    Example command for training preparation.
    """
    LOG.warning("Make sure you ran preprocess before this. Ignore if did")
    prepare_and_train(
        model_name,
        model_fold,
        gpu_id,
        dataset_id,
        gpu_space,
        email_address,
        determine_postprocessing,
        train_continue,
    )
