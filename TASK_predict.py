import shutil
import subprocess as sp
import time
from typing import List

from a9t.common.utils import remove_stuff
from a9t.config import MODEL_CONFIG
from a9t.dao.db import DBConnection
from a9t.dao.table import Status, DicomLog
from a9t.predict import folder_predict

GPU_RECHECK_TIME_SECONDS = 10
REQUIRED_FREE_MEMORY_BYTES = 6144
DEFAULT_PREDS_BASE_DIR = "output"


def get_free_gpu():
    memory_used_command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"]
    isolate_memory_value = lambda x: "".join(
        y for y in x.decode("ascii") if y in "0123456789"
    )
    mem_used = isolate_memory_value(
        sp.check_output(memory_used_command, stderr=sp.STDOUT)
    )

    return int(mem_used)


def copy_input_dcm_to_output(input_dir, output_dir):
    shutil.copytree(src=input_dir, dst=output_dir)
    remove_stuff(input_dir)


def send_to_external_server(pred_dcm_logs: List[DicomLog]):
    dcm_output_dirs = [dcm.output_path for dcm in pred_dcm_logs]
    # TODO: dcm_output_dirs got, check how to send to server
    pass


def run_prediction(seg_model_name):
    conn = DBConnection()
    all_dcm_files = conn.top(seg_model_name, Status.INIT)
    folder_predict(all_dcm_files, DEFAULT_PREDS_BASE_DIR, seg_model_name, True)
    pred_dcm_logs = conn.top(seg_model_name, Status.PREDICTED)
    for dcm in pred_dcm_logs:
        copy_input_dcm_to_output(dcm.input_path, dcm.output_path)
    send_to_external_server(pred_dcm_logs)


if __name__ == "__main__":
    while True:
        memory_used = get_free_gpu()
        if memory_used >= REQUIRED_FREE_MEMORY_BYTES:
            for model_name in MODEL_CONFIG["KEYS"]:
                run_prediction(model_name)
        else:
            time.sleep(GPU_RECHECK_TIME_SECONDS)
