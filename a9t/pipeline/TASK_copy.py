from functools import lru_cache
import glob
import os.path
from pathlib import Path
import time

from pydicom import dcmread
from watchdog.events import PatternMatchingEventHandler, FileSystemEvent
from watchdog.observers import Observer

from a9t.config import DCM_REGEX, DICOM_WATCH_DIR, PROTOCOL_TO_MODEL, LOG
from a9t.dao.db import DBConnection
from a9t.dao.table import DicomLog
from a9t.utils.debounce import debounce
from a9t.utils.ioutils import get_immediate_dicom_parent_dir

COPY_WAIT_SECONDS = 5
WATCH_DELAY = 1
RAW_DIR = os.path.join("data", "raw")
# Watchdog generates duplicate events. To fix that filter ROOT dir events
REDUNDANT_EVENT_PATH = Path(DICOM_WATCH_DIR)


def filter_files(path):
    # TODO: Change This Value
    # TODO: Use os.path.isfile check if needed
    return True


def determine_model(dir_path):
    model_name = None

    try:
        one_file_name = glob.glob(os.path.join(dir_path, DCM_REGEX), recursive=True)[0]
        ds = dcmread(one_file_name)
        dcm_protocol_name = ds.ProtocolName.lower()
        for protocol, model in PROTOCOL_TO_MODEL.items():
            if protocol in dcm_protocol_name:
                model_name = model
                break

    except Exception:
        LOG.error(f"Exception while processing: {dir_path}", exc_info=True)
    finally:
        return model_name


def on_modified(event: FileSystemEvent):
    src_path = Path(event.src_path)
    LOG.debug(f"Triggered for {src_path}")
    if event.is_directory and src_path.resolve() != REDUNDANT_EVENT_PATH.resolve():
        modification_event_trigger(event.src_path)


@lru_cache(maxsize=256)
def modification_event_trigger(src_path: str):
    LOG.info(f"MODIFIED {src_path}")

    dir_path = src_path
    wait_copy_finish(dir_path)
    model_name = determine_model(dir_path)
    if model_name is not None:
        conn = DBConnection()
        series_name = get_series_name(src_path)
        dcm = DicomLog(
            input_path=dir_path,
            model=model_name,
            series_name=series_name,
        )
        conn.insert([dcm])


def get_series_name(src_path):
    return os.path.basename(get_immediate_dicom_parent_dir(src_path))


def wait_copy_finish(filename):
    old_size = -1
    while old_size != os.path.getsize(filename):
        old_size = os.path.getsize(filename)
        time.sleep(COPY_WAIT_SECONDS)
    LOG.info(f"File {filename} copy complete detected")


def on_deleted(event):
    delete_event_trigger(event.src_path)


@lru_cache(maxsize=256)
def delete_event_trigger(src_path):
    LOG.info(f"DELETED {src_path}")


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
