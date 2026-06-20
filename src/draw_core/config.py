"""Runtime configuration for the core, passed explicitly (not read at import).

``CoreConfig`` carries the few tunables and paths the segmentation core needs.
Entrypoints build it once and inject it, so the core has no hidden global state
and no dependency on any particular config file format.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoreConfig:
    """Tunables for a segmentation run.

    nnU-Net data dirs default to the conventional ``data/...`` layout but are
    overridable per host/container/Batch job.
    """

    nnunet_raw_dir: str = "data/nnUNet_raw"
    nnunet_preprocessed_dir: str = "data/nnUNet_preprocessed"
    nnunet_results_dir: str = "data/nnUNet_results"

    # GPU scheduling — legacy defaults were tuned for an 11GB/4GB hospital GPU and
    # are deliberately conservative. Raise on big cloud/DGX GPUs (perf lever 1).
    required_free_gpu_mb: int = 5 * 1024
    pred_batch_size: int = 1

    # Inference behaviour
    disable_tta: bool = True
    checkpoint_name: str = "checkpoint_best.pth"
