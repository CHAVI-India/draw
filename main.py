import click

from a9t.config import ALL_SEG_MAP, MODEL_FOLDS
from a9t.dao.table import DicomLog
from a9t.predict import folder_predict
from a9t.preprocess.preprocess_data import run_pre_processing
from a9t.train.train import prepare_and_train
from a9t.utils.ioutils import get_all_dicom_dirs


@click.group(
    name="a9t",
    help="Use AutoSegment(A9T) pipeline for your needs. Use subcommands for your tasks",
)
def cli():
    pass


@cli.command(help="Preprocess DICOM Data to nnUNet format")
@click.option(
    "--root-dir",
    "-d",
    type=str,
    required=True,
    help="Parent Directory containing other DICOM directories",
)
@click.option(
    "--dataset-id",
    "-i",
    type=str,
    required=True,
    help="3 digit ID of the dataset",
)
@click.option(
    "--dataset-name",
    "-n",
    type=click.Choice(ALL_SEG_MAP.keys()),
    required=True,
    help="Name of the dataset from given list",
)
@click.option(
    "--start",
    "-s",
    type=int,
    required=False,
    default=0,
    help="The sample number to start putting data from",
)
@click.option(
    "--only-original",
    is_flag=True,
    help="Convert only original DICOM. Set this to disable RTStruct file searching and parsing",
)
def preprocess(
    root_dir: str,
    dataset_id: str,
    dataset_name: str,
    start: int,
    only_original: bool,
):
    run_pre_processing(dataset_id, dataset_name, only_original, root_dir, start)


@cli.command(help="Generate Predictions from trained model")
@click.option(
    "--preds-dir",
    "-p",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        writable=True,
    ),
    required=True,
    help="Output Directory that will contain final labels",
)
@click.option(
    "--root-dir",
    "-r",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        writable=True,
    ),
    required=True,
    help="Directory containing other DICOM parent directories",
)
@click.option(
    "--dataset-name",
    "-n",
    type=click.Choice(ALL_SEG_MAP.keys()),
    required=True,
    help="Name of the dataset",
)
@click.option(
    "--only-original",
    is_flag=True,
    help="Convert only original DICOM. Set this to disable RTStruct file searching and parsing",
)
def predict(preds_dir, dataset_name, root_dir, only_original):
    all_dicom_dirs = get_all_dicom_dirs(root_dir)
    dcm_logs = [DicomLog(input_path=input_path) for input_path in all_dicom_dirs]
    folder_predict(dcm_logs, preds_dir, dataset_name, only_original)


@cli.command(
    help="Prepare and Train model on a single GPU. Run Preprocessing Before running this"
)
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
def prep_train(
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


if __name__ == "__main__":
    cli()
