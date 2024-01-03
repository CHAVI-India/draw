import os
import zipfile

from draw.config import NNUNET_RESULTS_DATA_ENV_KEY, ALL_SEG_MAP, LOG


# @click.command()
# @click.option('--dataset-id', type=int, help='3-digit ID for the dataset', required=True)
# @click.option('--model-name', type=str, help='Model name', required=True)
def export_to_zip(dataset_id, model_name):
    """
    Create a ZIP archive with files from the specified dataset folder.
    """
    dataset_name = ALL_SEG_MAP[model_name]["models"][dataset_id]["name"]
    source_dir = (
        f"{os.environ[NNUNET_RESULTS_DATA_ENV_KEY]}/Dataset{dataset_id}_{dataset_name}"
    )

    # Check if the source directory exists
    if not os.path.exists(source_dir):
        LOG.error(f"Error: Source directory '{source_dir}' not found.")
        return

    # Create the ZIP file path
    zip_file_path = f"Dataset{dataset_id}_{dataset_name}.zip"

    # Create a ZIP archive
    with zipfile.ZipFile(zip_file_path, "w") as zipf:
        # Walk through the source directory and add files to the ZIP
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname=arc_name)

    LOG.info(f"ZIP file created successfully: {zip_file_path}")
