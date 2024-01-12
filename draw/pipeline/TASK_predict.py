from itertools import cycle
import shutil
import subprocess as sp
import time
from typing import List

from draw.config import MODEL_CONFIG, LOG, OUTPUT_DIR
from draw.dao.common import Status
from draw.dao.db import DBConnection
from draw.dao.table import DicomLog
from draw.predict import folder_predict
from retry.api import retry_call

GPU_RECHECK_TIME_SECONDS = 5
REQUIRED_FREE_MEMORY_BYTES = int(5 * 1024)


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return int(memory_free_values[0])


def copy_input_dcm_to_output(input_dir, output_dir):
    # This triggers WatchDog maybe?
    LOG.debug(f"Copying {input_dir} -> {output_dir}")
    shutil.copytree(src=input_dir, dst=output_dir, dirs_exist_ok=True)


def send_to_external_server(pred_dcm_logs: List[DicomLog]):
    # dcm_output_dirs = [dcm.output_path for dcm in pred_dcm_logs]
    # TODO: dcm_output_dirs got, check how to send to server
    for dcm in pred_dcm_logs:
        DBConnection.update_status_by_id(dcm, Status.SENT)
    LOG.info(f"Sent {len(pred_dcm_logs)} to server")


def run_prediction(seg_model_name):
    all_dcm_files = DBConnection.dequeue(seg_model_name)
    if len(all_dcm_files) > 0:
        run_prediction_with_retry(seg_model_name, all_dcm_files)
        pred_dcm_logs = DBConnection.top(seg_model_name, Status.PREDICTED)
        LOG.info(f"Got {len(pred_dcm_logs)} from DB")
        send_to_external_server(pred_dcm_logs)
        return True
    return False


def run_prediction_with_retry(seg_model_name, all_dcm_files):
    retry_call(
        folder_predict,
        fargs=(
            all_dcm_files,
            OUTPUT_DIR,
            seg_model_name,
            True,
        ),
        tries=2,
        logger=LOG,
    )


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
