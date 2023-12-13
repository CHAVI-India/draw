import click

from a9t.utils.ioutils import get_all_folders_from_raw_dir, get_all_dicom_dirs
from a9t.dao.table import DicomLog
from a9t.config import ALL_SEG_MAP, LOG, SAMPLE_NUMBER_ZFILL
from a9t.predict import folder_predict
from a9t.preprocess.preprocess_data import convert_dicom_dir_to_nnunet_dataset


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
def preprocess(root_dir, dataset_id, dataset_name, start, only_original):
    task_map = ALL_SEG_MAP[dataset_name]
    all_dicom_dirs = get_all_folders_from_raw_dir(root_dir)
    LOG.info("Processing ID", dataset_id)
    LOG.info(f"Found {len(all_dicom_dirs)} directories to work on...")

    dataset_specific_map = task_map[int(dataset_id)]
    dataset_name = dataset_specific_map["name"]
    seg_map = dataset_specific_map["map"]

    for idx, dicom_dir in enumerate(all_dicom_dirs, start=start):
        sample_number = str(idx).zfill(SAMPLE_NUMBER_ZFILL)

        convert_dicom_dir_to_nnunet_dataset(
            dicom_dir,
            dataset_id,
            dataset_name,
            sample_number,
            seg_map,
            data_tag="seg",
            extension="nii.gz",
            only_original=only_original,
        )


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


@cli.command(help="(V1) Train individual datasets")
def train_v1():
    raise NotImplemented("Processing...")


if __name__ == "__main__":
    cli()
