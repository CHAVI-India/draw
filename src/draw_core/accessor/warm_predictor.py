"""Warm, in-process nnU-Net predictor — the big no-retrain GPU speedups.

This is GPU perf lever 1 (warm resident model) + 2 (collapse repeated cold loads
into one) + 5 (torch.compile / cudnn.benchmark) + 8 (keep work on the GPU). It
holds an ``nnUNetPredictor`` per (dataset, config) resident for the process life,
so weights are loaded once instead of on every prediction.

Lazily imports torch/nnU-Net so importing ``draw_core`` stays GPU-free. Only used
on machines with the ``gpu`` extra installed; the subprocess path in
``NNUNetV2Adapter.predict_folder`` remains the backward-compatible default.

NOT enabled here: step_size tuning, FP16/TF32, or re-planning (levers 3/4/6) —
those change clinical Dice and need re-validation first.
"""

from __future__ import annotations

import os
from typing import Any

from draw_core.config import CoreConfig
from draw_core.logging import get_logger

log = get_logger(__name__)


class WarmPredictor:
    """Caches one nnUNetPredictor per model folder for the process lifetime."""

    def __init__(self, config: CoreConfig, gpu_id: int | None = None):
        self.config = config
        self.gpu_id = gpu_id
        self._cache: dict[str, Any] = {}
        self._tuned = False

    def _tune_backend_once(self) -> None:
        if self._tuned:
            return
        import torch

        # Lever 5: fixed patch sizes -> let cuDNN pick the fastest conv algorithm.
        torch.backends.cudnn.benchmark = True
        self._tuned = True

    def _model_folder(self, dataset_id: str, model_name: str, trainer_name: str,
                      model_config: str) -> str:
        return os.path.join(
            self.config.nnunet_results_dir,
            f"Dataset{dataset_id}_{model_name}",
            f"{trainer_name}__nnUNetPlans__{model_config}",
        )

    def _get_predictor(self, model_folder: str, fold: str):
        """Build-or-reuse a resident predictor for this model (lever 1 + 2)."""
        if model_folder in self._cache:
            return self._cache[model_folder]

        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        if self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)  # lever 7
        self._tune_backend_once()

        predictor = nnUNetPredictor(
            use_gaussian=True,
            use_mirroring=not self.config.disable_tta,
            perform_everything_on_device=True,  # lever 8
            device=torch.device("cuda"),
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(int(fold),),
            checkpoint_name=self.config.checkpoint_name,
        )
        log.info("Loaded resident predictor for %s (fold %s)", model_folder, fold)
        self._cache[model_folder] = predictor
        return predictor

    def predict_folder(
        self,
        samples_dir: str,
        output_dir: str,
        model_config: str,
        dataset_id: str,
        model_name: str,
        fold: str,
        trainer_name: str = "nnUNetTrainer",
    ) -> None:
        """Run inference with a resident predictor (no per-call model reload)."""
        os.makedirs(output_dir, exist_ok=True)
        model_folder = self._model_folder(dataset_id, model_name, trainer_name, model_config)
        predictor = self._get_predictor(model_folder, fold)
        predictor.predict_from_files(
            samples_dir,
            output_dir,
            save_probabilities=False,
            overwrite=True,
        )
