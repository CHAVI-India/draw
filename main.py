"""
CLI of AutoSegment (A9T) Pipeline
"""
import os
from datetime import datetime
import shutil

import click

from a9t.common.nifti2rt import convert_nifti_outputs_to_dicom
from a9t.common.utils import remove_stuff, get_all_folders_from_raw_dir
from a9t.mapping import ALL_SEG_MAP
from a9t.preprocess.preprocess_data import convert_dicom_dir_to_nnunet_dataset
from a9t.predict.evaluate import generate_labels_on_data
from a9t.postprocess import postprocess_folder


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
@click.option(
    "--only-original",
    is_flag=True,
    help="Convert only original DICOM",
)
def preprocess(root_dir, dataset_id, dataset_name, start, only_original):
    task_map = ALL_SEG_MAP[dataset_name]
    all_dicom_dirs = get_all_folders_from_raw_dir(root_dir)

    print("Processing ID", dataset_id)
    dataset_specific_map = task_map[int(dataset_id)]
    dataset_name = dataset_specific_map["name"]
    seg_map = dataset_specific_map["map"]

    for idx, dicom_dir in enumerate(all_dicom_dirs, start=start):
        sample_number = str(idx).zfill(3)

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
)
def predict(preds_dir, dataset_name, root_dir, only_original):
    task_map = ALL_SEG_MAP[dataset_name]
    all_dicom_dirs = get_all_folders_from_raw_dir(root_dir)
    exp_number = datetime.now().strftime("%Y-%m-%d.%H-%M")
    parent_dataset_name = dataset_name

    for dataset_id in task_map.keys():
        print("Processing ID", dataset_id)
        dataset_specific_map = task_map[dataset_id]
        model_config = dataset_specific_map["config"]
        dataset_name = dataset_specific_map["name"]
        seg_map = dataset_specific_map["map"]
        trainer_name = dataset_specific_map["trainer_name"]
        postprocess = dataset_specific_map["postprocess"]

        dataset_dir = f"data/nnUNet_raw/Dataset{dataset_id}_{dataset_name}"
        remove_stuff(dataset_dir)

        for idx, dicom_dir in enumerate(all_dicom_dirs):
            sample_number = str(idx).zfill(3)

            dataset_dir = convert_dicom_dir_to_nnunet_dataset(
                dicom_dir,
                dataset_id,
                dataset_name,
                sample_number,
                seg_map,
                data_tag="seg",
                extension="nii.gz",
                only_original=only_original,
            )
        tr_images = os.path.join(dataset_dir, "imagesTr")
        model_pred_dir = os.path.join(
            preds_dir, parent_dataset_name, str(dataset_id), "modelpred"
        )
        remove_stuff(model_pred_dir)

        generate_labels_on_data(
            tr_images, dataset_id, model_pred_dir, model_config, trainer_name
        )

        if postprocess is not None:
            ip_folder = model_pred_dir
            op_folder = os.path.join(
                preds_dir, parent_dataset_name, str(dataset_id), "postprocess"
            )
            remove_stuff(op_folder)
            os.makedirs(op_folder, exist_ok=True)
            pkl_file_src = postprocess
            pkl_file_dest = f"{op_folder}/postprocessing.pkl"
            # copy
            shutil.copy(pkl_file_src, pkl_file_dest)
            postprocess_folder(ip_folder, op_folder, pkl_file_dest)
            model_pred_dir = op_folder

        final_output_dir = os.path.join(preds_dir, parent_dataset_name, "results")
        convert_nifti_outputs_to_dicom(
            model_pred_dir,
            final_output_dir,
            dataset_dir,
            dataset_id,
            exp_number,
            seg_map,
        )


@cli.command()
def train():
    raise NotImplemented("Processing...")


if __name__ == "__main__":
    cli()
