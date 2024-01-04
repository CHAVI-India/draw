from multiprocessing import Process

from draw.config import LOG
from draw.pipeline.TASK_copy import task_watch_dir
from draw.pipeline.TASK_predict import task_model_prediction


def start_continuous_prediction():
    all_processes = []
    process_functions = [task_model_prediction, task_watch_dir]
    for fxn in process_functions:
        p = Process(target=fxn)
        LOG.info(f"Starting {fxn.__name__}")
        p.start()
        all_processes.append(p)
    for p in all_processes:
        p.join()
