"""Training routes through the TrainableEngine capability, not a direct adapter.

Verifies (CPU-only, no real nnU-Net): the nnU-Net engine's train() drives plan+train
through its adapter, and a non-trainable engine is correctly rejected by the
isinstance(engine, TrainableEngine) guard.
"""

from __future__ import annotations

from draw_core.config import CoreConfig
from draw_core.engines.base import TrainableEngine
from draw_core.engines.factory import build_engine
from draw_core.engines.nnunet import NnUNetEngine


def _nnunet_engine() -> NnUNetEngine:
    block = {
        "models": {
            720: {
                "name": "TSPrime",
                "config": "3d_fullres",
                "trainer_name": "nnUNetTrainerNoMirroring",
                "postprocess": None,
                "map": {1: "Bladder"},
            }
        }
    }
    return build_engine("nnunet", block, core_config=CoreConfig())


def test_nnunet_engine_is_trainable():
    assert isinstance(_nnunet_engine(), TrainableEngine)


def test_engine_train_drives_plan_then_train(monkeypatch):
    engine = _nnunet_engine()
    calls = []

    class FakeAdapter:
        def __init__(self, *a, **k):
            pass

        def plan(self, dataset_id, config, gpu_memory_gb=None):
            calls.append(("plan", dataset_id, config, gpu_memory_gb))

        def train(self, dataset_id, config, fold, trainer_name, resume, device_id):
            calls.append(("train", dataset_id, fold, trainer_name, resume, device_id))

    # The engine imports NNUNetV2Adapter lazily inside train(); patch at source.
    monkeypatch.setattr("draw_core.accessor.nnunetv2.NNUNetV2Adapter", FakeAdapter)

    engine.train("720", "0", gpu_space=8, device_id=1, train_continue=False,
                 determine_postprocessing=False)

    assert calls[0] == ("plan", "720", "3d_fullres", 8)
    assert calls[1] == ("train", "720", "0", "nnUNetTrainerNoMirroring", False, 1)


def test_non_trainable_engine_is_rejected():
    """A stranger engine without plan/train must not pass the training guard."""
    from draw_core.engines.factory import register_engine

    @register_engine("infer-only-test")
    def _build(engine_config, **ctx):
        class InferOnly:
            name = "infer-only-test"

            def segment(self, study_dirs, seg_map, work_dir):
                return []

        return InferOnly()

    engine = build_engine("infer-only-test", {})
    assert not isinstance(engine, TrainableEngine)
