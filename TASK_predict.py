import shutil
import subprocess as sp
import time
from typing import List

from a9t.common.utils import remove_stuff
from a9t.config import MODEL_CONFIG
from a9t.dao.db import DBConnection
from a9t.dao.table import Status, DicomLog
from a9t.predict import folder_predict

GPU_RECHECK_TIME_SECONDS = 30
REQUIRED_FREE_MEMORY_BYTES = 6 * 1024
DEFAULT_PREDS_BASE_DIR = "output"

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print("FREE GPU MEMORY:", memory_free_values)
    return int(memory_free_values[0])


def copy_input_dcm_to_output(input_dir, output_dir):
    shutil.copytree(src=input_dir, dst=output_dir, dirs_exist_ok=True)
    remove_stuff(input_dir)


def send_to_external_server(pred_dcm_logs: List[DicomLog]):
    dcm_output_dirs = [dcm.output_path for dcm in pred_dcm_logs]
    # TODO: dcm_output_dirs got, check how to send to server
    pass


def run_prediction(seg_model_name):
    conn = DBConnection()
    all_dcm_files = conn.top(seg_model_name, Status.INIT)
    if len(all_dcm_files) == conn.BATCH_SIZE:
        folder_predict(all_dcm_files, DEFAULT_PREDS_BASE_DIR, seg_model_name, True)
        pred_dcm_logs = conn.top(seg_model_name, Status.PREDICTED)
        for dcm in pred_dcm_logs:
            copy_input_dcm_to_output(dcm.input_path, dcm.output_path)
        send_to_external_server(pred_dcm_logs)
        return True
    return False



if __name__ == "__main__":
    while True:
        gpu_memory_free = get_gpu_memory()
        if gpu_memory_free >= REQUIRED_FREE_MEMORY_BYTES:
            print("STARTING PREDICTIING...")
            model_ran = False
            for model_name in MODEL_CONFIG["KEYS"]:
                model_ran = model_ran or run_prediction(model_name)

        if not model_ran:
            print("GPU NOT FREE")
            time.sleep(GPU_RECHECK_TIME_SECONDS)
