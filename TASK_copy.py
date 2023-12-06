import glob
import os.path
import shutil
import time

from watchdog.events import PatternMatchingEventHandler, FileSystemEvent
from watchdog.observers import Observer

from a9t.common.utils import get_immediate_dicom_parent_dir
from a9t.dao.db import DBConnection
from a9t.dao.table import DicomLog

SERVER_OUTPUT_DIR = os.path.join("data", "watch")
DICOM_FILE_FILTER_REGEX = "**/*.dcm"
WAIT_FOR_COPY_PAUSE_SECONDS = 1
RAW_DIR = os.path.join("data", "raw")


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
    # TODO: Change this value
    model_name = "TSPrime"
    return model_name, os.path.join(RAW_DIR, model_name)


def on_modified(event: FileSystemEvent):
    print("New Event Detected")

    if event.is_directory:
        dir_path = event.src_path
        wait_copy_finish(dir_path)
        model_name, raw_dir = determine_model(dir_path)
        dir_path = copy_filtered_files(dir_path, raw_dir, filter_files)
        print("Updated Dir Path", dir_path)

        conn = DBConnection()
        series_name = os.path.basename(get_immediate_dicom_parent_dir(event.src_path))
        dcm = DicomLog(
            input_path=dir_path,
            model=model_name,
            series_name=series_name,
        )
        conn.insert([dcm])
        print(f"Added {dir_path} in DB")


def wait_copy_finish(filename):
    historicalSize = -1
    while historicalSize != os.path.getsize(filename):
        historicalSize = os.path.getsize(filename)
        time.sleep(1)
    print("file copy has now finished")


def on_deleted(event):
    print(f"DELETED {event.src_path}!")


if __name__ == "__main__":
    patterns = ["*"]
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(
        patterns, ignore_patterns, ignore_directories, case_sensitive
    )
    my_event_handler.on_modified = on_modified
    my_event_handler.on_deleted = on_deleted

    path = os.path.normpath(SERVER_OUTPUT_DIR)
    print("WATCHING", path)
    go_recursively = False
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)
    my_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()
