"""Engine factory: registration, resolution, defaults, and clear failure."""

from __future__ import annotations

import pytest

from draw_core.engines.base import SegmentationEngine, TrainableEngine
from draw_core.engines.factory import (
    available_engines,
    build_engine,
    register_engine,
)


def test_nnunet_is_registered_by_default():
    assert "nnunet" in available_engines()


def test_none_engine_defaults_to_nnunet():
    engine = build_engine(None, {"models": {}})
    assert engine.name == "nnunet"


def test_unknown_engine_fails_clearly():
    with pytest.raises(ValueError, match="Unknown engine"):
        build_engine("does-not-exist", {})


def test_nnunet_engine_satisfies_protocols():
    engine = build_engine("nnunet", {"models": {}})
    assert isinstance(engine, SegmentationEngine)
    assert isinstance(engine, TrainableEngine)  # nnU-Net is trainable


def test_a_stranger_engine_registers_and_builds():
    @register_engine("dummy-test-engine")
    def _build(engine_config, **ctx):
        class DummyEngine:
            name = "dummy-test-engine"

            def segment(self, study_dirs, seg_map, work_dir):
                return []

        return DummyEngine()

    engine = build_engine("dummy-test-engine", {"anything": 1})
    assert isinstance(engine, SegmentationEngine)
    assert not isinstance(engine, TrainableEngine)  # no plan/train -> not trainable
