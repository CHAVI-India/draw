import json
import time
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

from a9t.adapters.nnunetv2 import NNUNetV2Adapter, default_nnunet_adapter
from a9t.constants import MODEL_CONFIG, DEFAULT_FOLD

DATASET_JSON_FILENAME = "dataset.json"
PLANS_JSON_FILENAME = "plans.json"
SUMMARY_JSON_FILENAME = "summary.json"


def evaluate_nnunet_on_folder(
    labels_dir: str,
    preds_dir: str,
    nnunet_adapter: NNUNetV2Adapter,
) -> dict:
    """
    Evaluates NNUNet on folder to generate summary.json file,
    containing scores per sample
    Args:
        nnunet_adapter: Adapter to access NNUNet
        labels_dir (str): dir containing the original labels created while preprocessing. No trailing slash
        preds_dir (str): dir containing the predictions from the model. No trailing slash

    Returns:
        dict, The ``summary.json`` file as dict containing per sample scores

    """
    dj_file = f"{preds_dir}/{DATASET_JSON_FILENAME}"
    p_file = f"{preds_dir}/{PLANS_JSON_FILENAME}"
    nnunet_adapter.evaluate_on_folder(labels_dir, preds_dir, dj_file, p_file)

    with open(f"{preds_dir}/{SUMMARY_JSON_FILENAME}", "r") as fp:
        s_file = json.load(fp)
        # As per nnunet
        return s_file["metric_per_case"]


def convert_nifti_labels_to_predictions(
    labels_dir: str,
    preds_dir: str,
):
    pass


def generate_labels_on_data(
    samples_dir,
    dataset_id,
    output_dir,
    nnunet_adapter: NNUNetV2Adapter = default_nnunet_adapter,
):
    # start = time.time()
    # print("RT START:", start)
    start = datetime.now(IST)
    print("RT START:", start)

    nnunet_adapter.predict_folder(
        samples_dir,
        output_dir,
        MODEL_CONFIG,
        dataset_id,
        DEFAULT_FOLD,
    )
    # stop = time.time()
    # print("RT STOP:", stop)

    print("-"*50)
    stop = datetime.now(IST)
    print("RT START:", start)
    print("RT STOP:", stop)
    print("-"*50)
    
    # time_diff = (stop - start)
    # print("RT PRED DONE. TIME TAKEN:", stop - start)


def generate_final_predicitons():
    pass
