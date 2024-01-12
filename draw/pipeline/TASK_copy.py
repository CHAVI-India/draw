import glob
import os.path
from pathlib import Path
import time

from pydicom import dcmread
from watchdog.events import PatternMatchingEventHandler, FileSystemEvent
from watchdog.observers import Observer

from draw.config import (
    DCM_REGEX,
    DICOM_WATCH_DIR,
    PROTOCOL_TO_MODEL,
    LOG,
    DicomKeyToTag,
)
from draw.dao.common import Status
from draw.dao.db import DBConnection
from draw.dao.table import DicomLog
from draw.utils.ioutils import get_dicom_attribute_from_dir

COPY_WAIT_SECONDS = 20
WATCH_DELAY = 1
# Watchdog generates duplicate events. To fix that filter ROOT dir events
REDUNDANT_EVENT_PATH = Path(DICOM_WATCH_DIR).resolve()


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

    except IndexError:
        LOG.error(f"No DCM found. Probably spurious event", exc_info=True)
    except AttributeError:
        LOG.error(f"Protocol Not Found", exc_info=True)
    except Exception:
        LOG.error(f"Ignored Exception while processing: {dir_path}", exc_info=True)
    finally:
        return model_name


def on_modified(event: FileSystemEvent):
    src_path = Path(event.src_path)
    if (
        event.is_directory
        and src_path.resolve() != REDUNDANT_EVENT_PATH
        and not event.is_synthetic
    ):
        modification_event_trigger(event.src_path)


def modification_event_trigger(src_path: str):
    LOG.info(f"MODIFIED {src_path}")

    try:
        dir_path = src_path
        series_name = get_uniq_id_for_sample(src_path)

        if (
            series_name is None
            or not os.path.exists(dir_path)
            or DBConnection.exists(series_name)
        ):
            LOG.info(f"Duplicate Event detected @ {src_path} with series {series_name}")
            return

        wait_copy_finish(dir_path)
        model_name = determine_model(dir_path)

        if model_name is not None:
            dcm = DicomLog(
                input_path=dir_path,
                model=model_name,
                series_name=series_name,
                status=Status.INIT,
            )
            DBConnection.enqueue([dcm])
        else:
            LOG.warning(f"SRC {src_path} not processed as no Valid model found")

    except IndexError:
        LOG.error(f"Probably Spurious Event from OS", exc_info=True)

    except Exception:
        LOG.error(f"Error while processing modification {src_path}", exc_info=True)


def get_uniq_id_for_sample(src_path):
    return get_dicom_attribute_from_dir(src_path, DicomKeyToTag.series_instance_uid)


def wait_copy_finish(filename):
    old_size = -1
    while old_size != os.path.getsize(filename):
        old_size = os.path.getsize(filename)
        time.sleep(COPY_WAIT_SECONDS)
    LOG.info(f"File {filename} copy complete detected")


def on_deleted(event):
    delete_event_trigger(event.src_path)


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
