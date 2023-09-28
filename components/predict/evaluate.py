import json
import subprocess
import sys

DATASET_JSON_FILENAME = "dataset.json"
PLANS_JSON_FILENAME = "plans.json"
SUMMARY_JSON_FILENAME = "summary.json"


def evaluate_nnunet_on_folder(
    labels_dir: str,
    preds_dir: str,
) -> dict:
    """
    Evaluates NNUNet on folder to generate summary.json file,
    containing scores per sample
    Args:
        labels_dir (str): dir containing the original labels created while preprocessing. No trailing slash
        preds_dir (str): dir containing the predictions from the model. No trailing slash

    Returns:
        dict, The ``summary.json`` file as dict containing per sample scores

    """
    dj_file = f"{preds_dir}/{DATASET_JSON_FILENAME}"
    p_file = f"{preds_dir}/{PLANS_JSON_FILENAME}"
    # Synchronous call, waits for subprocess to finish
    subprocess.run(
        [
            "nnUNetv2_evaluate_folder",
            labels_dir,
            preds_dir,
            "-djfile",
            dj_file,
            "-pfile",
            p_file,
            "--chill",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )

    with open(f"{preds_dir}/{SUMMARY_JSON_FILENAME}", "r") as fp:
        s_file = json.load(fp)
        # As per nnunet
        return s_file["metric_per_case"]

def convert_nifti_labels_to_predictions(
        labels_dir: str,
        preds_dir: str,
):



