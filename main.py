"""
CLI of AutoSegment (A9T) Pipeline
"""
import os

import click

from a9t.adapters.nnunetv2 import default_nnunet_adapter
from a9t.class_mapping import ALL_SEG_MAP
from a9t.converters.nifti2rt import convert_nifti_outputs_to_dicom
from a9t.predict.evaluate import (
    generate_labels_on_data,
)
from a9t.preprocess.preprocess_data import (
    convert_dicom_dir_to_nnunet_dataset,
    nnunet_preprocess,
    clear_old_training_data,
)


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
    clear_old_training_data(dataset_id, dataset_name)
    all_dicom_dirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    print(len(all_dicom_dirs))

    for idx, dicom_dir in enumerate(all_dicom_dirs, start=start):
        sample_number = str(idx).zfill(3)

        convert_dicom_dir_to_nnunet_dataset(
            dicom_dir,
            dataset_id,
            dataset_name,
            sample_number,
            only_original=False,
            data_tag="seg",
            extension="nii.gz",
        )
    nnunet_preprocess(dataset_id, default_nnunet_adapter)


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
    help="Output Directory",
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
    help="Directory containing Raw Extracted Data of the dataset",
)
@click.option(
    "--dataset-name",
    "-n",
    type=click.Choice(ALL_SEG_MAP.keys()),
    required=True,
    help="Name of the dataset",
)
@click.option(
    "--dataset-id",
    "-d",
    type=str,
    required=True,
    help="ID of the dataset",
)
@click.option("--only-original", is_flag=True, help="Convert only original or both DICOM and RT")
def predict(preds_dir, dataset_id, dataset_name, root_dir, only_original):
    clear_old_training_data(dataset_id, dataset_name)
    all_dicom_dirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]

    dataset_dir = None

    for idx, dicom_dir in enumerate(all_dicom_dirs):
        sample_number = str(idx).zfill(3)
        print("Currently at", dicom_dir)

        dataset_dir = convert_dicom_dir_to_nnunet_dataset(
            dicom_dir,
            dataset_id,
            dataset_name,
            sample_number,
            only_original=only_original,
            data_tag="seg",
            extension="nii.gz",
        )
    generate_labels_on_data(f"{dataset_dir}/imagesTr", dataset_id, preds_dir)
    convert_nifti_outputs_to_dicom(preds_dir, dataset_dir, dataset_id, dataset_name)


@cli.command()
def train():
    raise NotImplemented("Processing...")


if __name__ == "__main__":
    cli()
