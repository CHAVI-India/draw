import glob
import os.path
from pathlib import Path
import shutil
import time

from pydicom import dcmread
from watchdog.events import PatternMatchingEventHandler, FileSystemEvent
from watchdog.observers import Observer

from a9t.config import DICOM_WATCH_DIR, PROTOCOL_TO_MODEL, LOG
from a9t.dao.db import DBConnection
from a9t.dao.table import DicomLog, Status
from a9t.utils.ioutils import get_immediate_dicom_parent_dir
import threading


COPY_WAIT_SECONDS = 5
DICOM_FILE_FILTER_REGEX = "**/*.dcm"
WATCH_DELAY = 1
RAW_DIR = os.path.join("data", "raw")
# Watchdog generates duplicate events
REDUNDANT_EVENT_PATH = Path(DICOM_WATCH_DIR)


def debounce(wait_time):
    """
    Decorator that will debounce a function so that it is called after wait_time seconds
    If it is called multiple times, will wait for the last call to be debounced and run only this one.
    """

    def decorator(function):
        def debounced(*args, **kwargs):
            def call_function():
                debounced._timer = None
                return function(*args, **kwargs)

            # if we already have a call to the function currently waiting to be executed, reset the timer
            if debounced._timer is not None:
                debounced._timer.cancel()

            # after wait_time, call the function provided to the decorator with its arguments
            debounced._timer = threading.Timer(wait_time, call_function)
            debounced._timer.start()

        debounced._timer = None
        return debounced

    return decorator


def copy_filtered_files(src_dir, dst_base_dir, filter_fxn):
    updated_dir_path = None
    dcm_parent_dir = get_immediate_dicom_parent_dir(src_dir)
    for p in glob.glob(DICOM_FILE_FILTER_REGEX, recursive=True, root_dir=src_dir):
        if filter_fxn(os.path.join(src_dir, p)):
            updated_dir_path = os.path.join(
                dst_base_dir, os.path.basename(dcm_parent_dir)
            )
            os.makedirs(updated_dir_path, exist_ok=True)
            shutil.copy(os.path.join(src_dir, p), os.path.join(dst_base_dir, p))
    return updated_dir_path


def filter_files(path):
    # TODO: Change This Value
    # TODO: Use os.path.isfile check if needed
    return True


def determine_model(dir_path):
    model_name = None

    try:
        one_file_name = glob.glob(
            os.path.join(dir_path, DICOM_FILE_FILTER_REGEX), recursive=True
        )[0]
        ds = dcmread(one_file_name)
        dcm_protocol_name = ds.ProtocolName.lower()
        for protocol, model in PROTOCOL_TO_MODEL.items():
            if protocol in dcm_protocol_name:
                model_name = model
                break

        if model_name is None:
            return None, None

        return model_name, os.path.join(RAW_DIR, model_name)

    except Exception:
        LOG.error(f"Exception while processing: {dir_path}", exc_info=True)
        return None, None


@debounce(5)
def on_modified(event: FileSystemEvent):
    path = Path(event.src_path)

    if path.resolve() != REDUNDANT_EVENT_PATH.resolve():
        LOG.info(f"Modification Detected at {event.src_path}")

        if event.is_directory:
            dir_path = event.src_path
            wait_copy_finish(dir_path)
            model_name, _ = determine_model(dir_path)
            if model_name is not None:
                LOG.info(f"Dir path {dir_path}")

                conn = DBConnection()
                series_name = os.path.basename(
                    get_immediate_dicom_parent_dir(event.src_path)
                )
                dcm = DicomLog(
                    input_path=dir_path,
                    model=model_name,
                    series_name=series_name,
                    status=Status.INIT,
                )
                conn.insert([dcm])
                LOG.info(f"{dir_path} in DB with INIT")


def wait_copy_finish(filename):
    old_size = -1
    while old_size != os.path.getsize(filename):
        old_size = os.path.getsize(filename)
        time.sleep(COPY_WAIT_SECONDS)
    LOG.info(f"File {filename} copy complete detected")


@debounce(5)
def on_deleted(event):
    LOG.info(f"DELETED {event.src_path}!")


def task_watch_dir():
    patterns = ["*"]
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(
        patterns, ignore_patterns, ignore_directories, case_sensitive
    )
    my_event_handler.on_modified = on_modified
    my_event_handler.on_deleted = on_deleted
    path = os.path.normpath(DICOM_WATCH_DIR)

    LOG.info(f"Started watching {path} for modifications")

    go_recursively = False
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)
    my_observer.start()
    try:
        while True:
            time.sleep(WATCH_DELAY)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()
