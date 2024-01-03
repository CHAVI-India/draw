from itertools import cycle
import shutil
import subprocess as sp
import time
from typing import List

from draw.config import MODEL_CONFIG, LOG
from draw.dao.db import DBConnection
from draw.dao.table import Status, DicomLog
from draw.predict import folder_predict

GPU_RECHECK_TIME_SECONDS = 15
REQUIRED_FREE_MEMORY_BYTES = int(5 * 1024)
DEFAULT_PREDS_BASE_DIR = "output"


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return int(memory_free_values[0])


def copy_input_dcm_to_output(input_dir, output_dir):
    shutil.copytree(src=input_dir, dst=output_dir, dirs_exist_ok=True)


def send_to_external_server(pred_dcm_logs: List[DicomLog]):
    # dcm_output_dirs = [dcm.output_path for dcm in pred_dcm_logs]
    # TODO: dcm_output_dirs got, check how to send to server
    conn = DBConnection()
    for dcm in pred_dcm_logs:
        conn.update_status(dcm, Status.SENT)


def run_prediction(seg_model_name):
    conn = DBConnection()
    all_dcm_files = conn.top(seg_model_name, Status.INIT)
    if len(all_dcm_files) > 0:
        folder_predict(all_dcm_files, DEFAULT_PREDS_BASE_DIR, seg_model_name, True)
        pred_dcm_logs = conn.top(seg_model_name, Status.PREDICTED)
        for dcm in pred_dcm_logs:
            copy_input_dcm_to_output(dcm.input_path, dcm.output_path)
        send_to_external_server(pred_dcm_logs)
        return True
    return False


def task_model_prediction():
    # Python 3.8 minimum for this operator
    model_name_generator = cycle(MODEL_CONFIG["KEYS"])
    while model_name := next(model_name_generator):
        try:
            gpu_memory_free = get_gpu_memory()
            any_model_ran = False

            if gpu_memory_free >= REQUIRED_FREE_MEMORY_BYTES:
                LOG.info(f"{gpu_memory_free} MB free GPU. Trying {model_name}")
                any_model_ran = any_model_ran or run_prediction(model_name)

            if not any_model_ran:
                LOG.info(f"{model_name} ran: {any_model_ran}")
                time.sleep(GPU_RECHECK_TIME_SECONDS)
        except Exception:
            LOG.error("Exception Ignored", exc_info=True)
            continue
