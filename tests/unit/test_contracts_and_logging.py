"""Unit tests for the contract types and the logging helpers."""

from __future__ import annotations

import logging

from draw_contracts import JobStatus, ModelSpec, NullStatusSink, SegmentationJob
from draw_contracts.protocols import StatusSink
from draw_core.logging import configure_logging, get_logger


def test_null_sink_satisfies_protocol():
    sink = NullStatusSink()
    assert isinstance(sink, StatusSink)
    # No-ops, no exceptions.
    sink.record_predicted("series-1", "/out/series-1")
    sink.record_failed("job-1", "boom")


def test_job_dto_is_frozen():
    job = SegmentationJob(
        job_id="j1", input_uri="/in", output_uri="/out", model=ModelSpec("TSPrime")
    )
    assert job.model.only_original is True
    assert JobStatus.FAILED.value == "FAILED"


def test_configure_logging_is_idempotent_and_streams(capsys):
    configure_logging(level="INFO")
    configure_logging(level="INFO")  # second call must not duplicate handlers
    managed = [h for h in logging.getLogger().handlers if getattr(h, "_draw_managed", False)]
    assert len(managed) == 1

    get_logger("draw.test").info("hello-functional")
    assert "hello-functional" in capsys.readouterr().out


def test_configure_logging_writes_file(tmp_path):
    logfile = tmp_path / "nested" / "draw.log"
    configure_logging(level="INFO", logfile=str(logfile), stream=False)
    get_logger("draw.test").info("to-file")
    for h in logging.getLogger().handlers:
        h.flush()
    assert logfile.exists()
    assert "to-file" in logfile.read_text()


def test_file_handler_uses_retention_and_gzip_rotation(tmp_path):
    import logging.handlers

    from draw_core.logging import _gzip_rotator

    logfile = tmp_path / "draw.log"
    configure_logging(
        level="INFO", logfile=str(logfile), stream=False, retention_days=30, compress=True
    )
    handlers = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.handlers.TimedRotatingFileHandler)
    ]
    assert len(handlers) == 1
    fh = handlers[0]
    assert fh.backupCount == 30          # 30-day retention
    assert fh.when == "MIDNIGHT"         # daily rotation
    assert fh.rotator is _gzip_rotator   # rotated files are gzipped


def test_gzip_rotator_compresses_and_removes_source(tmp_path):
    import gzip

    from draw_core.logging import _gzip_rotator

    src = tmp_path / "draw.log.2026-01-01"
    src.write_text("yesterday")
    _gzip_rotator(str(src), str(src))

    assert not src.exists()                          # original removed
    gz = tmp_path / "draw.log.2026-01-01.gz"
    assert gz.exists()
    assert gzip.open(gz, "rt").read() == "yesterday"  # content preserved, compressed
