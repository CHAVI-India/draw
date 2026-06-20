"""Watcher task: detect new DICOM study dirs and enqueue them.

Ported from ``draw/pipeline/TASK_copy.py``. Dependencies (DB connection, the
protocol->model lookup, watch dir) are passed in rather than read from import-time
globals.

KNOWN BUGS (deliberately left, tracked separately — out of scope for this port):
  * ``wait_copy_finish`` calls ``os.path.getsize`` on a *directory*, which does not
    reflect a directory's growing contents; copy-completion detection is unreliable.
    TODO(draw-pipeline): replace with a stable mtime/size sweep over the dir tree.
  * watchdog emits duplicate/synthetic events; only coarse filtering is applied.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path

from pydicom import dcmread
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer

from draw_contracts.queue import JobQueue
from draw_core.constants import DCM_REGEX, DICOM_TAG_SERIES_INSTANCE_UID
from draw_core.logging import get_logger
from draw_core.models import ModelRegistry

log = get_logger(__name__)

COPY_WAIT_SECONDS = 20
WATCH_DELAY = 1


def determine_model(dir_path: str, registry: ModelRegistry) -> str | None:
    try:
        one_file_name = glob.glob(os.path.join(dir_path, DCM_REGEX), recursive=True)[0]
        ds = dcmread(one_file_name)
        dcm_protocol_name = ds.ProtocolName.lower()
        for cfg in registry.configs.values():
            if cfg.protocol in dcm_protocol_name:
                return cfg.name
    except IndexError:
        log.error("No DCM found. Probably spurious event", exc_info=True)
    except AttributeError:
        log.error("Protocol not found", exc_info=True)
    except Exception:
        log.error("Ignored exception while processing: %s", dir_path, exc_info=True)
    return None


def _series_uid_from_dir(dir_path: str) -> str | None:
    files = glob.glob(os.path.join(dir_path, DCM_REGEX), recursive=True)
    if not files:
        return None
    ds = dcmread(files[0])
    if DICOM_TAG_SERIES_INSTANCE_UID in ds:
        return str(ds[DICOM_TAG_SERIES_INSTANCE_UID].value)
    return None


def wait_copy_finish(filename: str) -> None:
    # TODO(draw-pipeline): getsize on a directory is unreliable; see module docstring.
    old_size = -1
    while old_size != os.path.getsize(filename):
        old_size = os.path.getsize(filename)
        time.sleep(COPY_WAIT_SECONDS)
    log.info("File %s copy complete detected", filename)


def modification_event_trigger(src_path: str, queue: JobQueue, registry: ModelRegistry) -> None:
    log.info("MODIFIED %s", src_path)
    try:
        series_name = _series_uid_from_dir(src_path)
        if series_name is None or not os.path.exists(src_path) or queue.exists(series_name):
            log.info("Duplicate event @ %s with series %s", src_path, series_name)
            return

        wait_copy_finish(src_path)
        model_name = determine_model(src_path, registry)
        if model_name is not None:
            queue.enqueue(series_name=series_name, input_path=src_path, model=model_name)
        else:
            log.warning("SRC %s not processed as no valid model found", src_path)
    except IndexError:
        log.error("Probably spurious event from OS", exc_info=True)
    except Exception:
        log.error("Error while processing modification %s", src_path, exc_info=True)


def task_watch_dir(watch_dir: str, queue: JobQueue, registry: ModelRegistry) -> None:
    path = os.path.normpath(watch_dir)
    redundant_event_path = Path(path).resolve()

    def on_modified(event: FileSystemEvent) -> None:
        src_path = Path(event.src_path)
        if (
            event.is_directory
            and src_path.resolve() != redundant_event_path
            and not event.is_synthetic
        ):
            modification_event_trigger(event.src_path, queue, registry)

    def on_deleted(event: FileSystemEvent) -> None:
        log.info("DELETED %s", event.src_path)

    handler = PatternMatchingEventHandler(["*"], None, False, True)
    handler.on_modified = on_modified
    handler.on_deleted = on_deleted

    log.info("Started watching %s for modifications", path)
    observer = Observer()
    observer.schedule(handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(WATCH_DELAY)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
