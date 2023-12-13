from multiprocessing import Process

from a9t.config import LOG
from a9t.pipeline.TASK_copy import task_watch_dir
from a9t.pipeline.TASK_predict import task_model_prediction

if __name__ == "__main__":
    all_processes = []
    process_functions = [task_model_prediction, task_watch_dir]

    for fxn in process_functions:
        p = Process(target=fxn)
        LOG.info(f"Starting {fxn.__name__}")
        p.start()
        all_processes.append(p)

    for p in all_processes:
        p.join()
