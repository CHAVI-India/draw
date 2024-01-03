import click

from draw.config import ALL_SEG_MAP
from draw.preprocess.preprocess_data import run_pre_processing


@click.command(help="Preprocess DICOM Data to nnUNet format")
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
def cli_preprocess(
    root_dir: str,
    dataset_id: str,
    dataset_name: str,
    start: int,
    only_original: bool,
):
    run_pre_processing(dataset_id, dataset_name, only_original, root_dir, start)
