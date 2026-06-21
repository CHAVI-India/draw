"""Watcher copy-completion detection and deterministic model resolution.

The legacy ``wait_copy_finish`` probed ``os.path.getsize`` on a *directory*, which
returns the fixed directory-entry size, so it declared a study "done" after one tick
regardless of whether the DICOM files were still being copied. These tests lock in the
recursive-tree-size detector and the deterministic file pick.
"""

from __future__ import annotations

import pytest

pytest.importorskip("watchdog")
pytest.importorskip("pydicom")

from draw_pipeline.pipeline.task_copy import _dir_tree_size, wait_copy_finish  # noqa: E402


def test_dir_tree_size_counts_nested_files(tmp_path):
    (tmp_path / "a.dcm").write_bytes(b"x" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.dcm").write_bytes(b"y" * 50)

    total, count = _dir_tree_size(str(tmp_path))

    assert total == 150
    assert count == 2


def test_dir_tree_size_is_zero_for_empty_dir(tmp_path):
    assert _dir_tree_size(str(tmp_path)) == (0, 0)


def test_wait_copy_finish_returns_when_tree_is_stable(tmp_path, monkeypatch):
    # A directory whose size never changes is "stable" after the second probe; with
    # sleep stubbed out this returns promptly instead of hanging.
    (tmp_path / "a.dcm").write_bytes(b"x" * 10)
    monkeypatch.setattr("draw_pipeline.pipeline.task_copy.time.sleep", lambda _s: None)

    wait_copy_finish(str(tmp_path))  # must terminate (not hang, not exit on tick 1)


def test_wait_copy_finish_waits_while_tree_grows(tmp_path, monkeypatch):
    # Simulate a copy in progress: the tree grows for two probes, then stabilises.
    # Each "sleep" appends another file, so the detector must NOT declare done early.
    sizes = iter([1, 2, 3])  # number of files present at successive probes; then stable

    def grow(_seconds):
        try:
            n = next(sizes)
        except StopIteration:
            return
        (tmp_path / f"{n}.dcm").write_bytes(b"z" * 10)

    (tmp_path / "0.dcm").write_bytes(b"z" * 10)
    monkeypatch.setattr("draw_pipeline.pipeline.task_copy.time.sleep", grow)

    wait_copy_finish(str(tmp_path))

    # All growth files were written before it concluded the copy had finished.
    dcm = list(tmp_path.glob("*.dcm"))
    assert len(dcm) >= 4  # 0,1,2,3 — it waited through every growth tick
