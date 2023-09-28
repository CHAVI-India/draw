"""
CLI of AutoSegment (A9T) Pipeline
"""
import os

import click

from components.class_mapping import ALL_SEG_MAP
from components.predict.evaluate import evaluate_nnunet_on_folder
from components.preprocess.preprocess_data import convert_dicom_dir_to_nnunet_dataset


@click.group(
    name="a9t",
    help="Use AutoSegment(A9T) pipeline for your needs. Use subcommands for your tasks",
)
def cli():
    pass


@cli.command(help="Convert DICOM Data to NNUNet format")
@click.option(
    "--root-dir",
    "-d",
    type=str,
    required=True,
    help="Parent Directory containing extracted DICOM directories",
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
    help="Name of the dataset",
)
@click.option(
    "--start",
    "-s",
    type=int,
    required=False,
    default=0,
    help="The sample number to start putting data from",
)
def preprocess(root_dir, dataset_id, dataset_name, start):
    all_dicom_dirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]

    for idx, dicom_dir in enumerate(all_dicom_dirs, start=start):
        sample_number = str(idx).zfill(3)

        convert_dicom_dir_to_nnunet_dataset(
            dicom_dir,
            dataset_id,
            dataset_name,
            sample_number,
            data_tag="seg",
            extension="nii.gz",
        )


@cli.command(help="Generate Predictions from trained model (TODO: incomplete)")
@click.option(
    "--labels-dir",
    "-l",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    required=True,
    help="Directory containing original labels of the dataset",
)
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
    help="Directory containing predicted labels of the dataset",
)
def predict(labels_dir, preds_dir):
    evaluate_nnunet_on_folder(labels_dir, preds_dir)


@cli.command()
def train():
    pass


if __name__ == "__main__":
    cli()
