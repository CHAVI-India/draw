"""Predictor seam: a uniform inference interface over a SubModel.

``segment_study`` builds ONE predictor and reuses it across every submodel of a
study. Two implementations satisfy the same shape:

* ``SubprocessPredictor`` — wraps ``NNUNetV2Adapter`` and shells out to the
  ``nnUNetv2_predict`` CLI. Backward-compatible default; cold-loads weights per call.
* ``WarmPredictor`` (see ``warm_predictor.py``) — keeps an in-process
  ``nnUNetPredictor`` resident, so weights load once (GPU perf levers 1, 2, 5, 8).

Taking a typed ``SubModel`` (not loose string args) keeps the call sites clean.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from draw_core.constants import DEFAULT_FOLD
from draw_core.models import SubModel


@runtime_checkable
class Predictor(Protocol):
    """Runs nnU-Net inference for one submodel into ``output_dir``."""

    def predict_submodel(self, submodel: SubModel, samples_dir: str, output_dir: str) -> None: ...


class SubprocessPredictor:
    """Cold-start CLI predictor. Wraps an ``NNUNetV2Adapter``. Default path."""

    def __init__(self, adapter, fold: str = DEFAULT_FOLD):
        self.adapter = adapter
        self.fold = fold

    def predict_submodel(self, submodel: SubModel, samples_dir: str, output_dir: str) -> None:
        self.adapter.predict_folder(
            samples_dir,
            output_dir,
            submodel.config,
            str(submodel.dataset_id),
            self.fold,
            submodel.trainer_name,
        )
