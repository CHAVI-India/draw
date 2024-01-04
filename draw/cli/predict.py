import click

from draw.config import ALL_SEG_MAP
from draw.dao.table import DicomLog
from draw.predict import folder_predict
from draw.utils.ioutils import get_all_dicom_dirs


@click.command(help="Generate Predictions from trained model")
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
def cli_predict(preds_dir, dataset_name, root_dir, only_original):
    all_dicom_dirs = get_all_dicom_dirs(root_dir)
    dcm_logs = [DicomLog(input_path=input_path) for input_path in all_dicom_dirs]
    folder_predict(dcm_logs, preds_dir, dataset_name, only_original)
