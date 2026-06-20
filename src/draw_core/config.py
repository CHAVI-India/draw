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

    # GPU perf levers 1/2/5/8: when True, segment_study uses an in-process resident
    # nnUNetPredictor (weights loaded once, reused across submodels and studies)
    # instead of the cold-start subprocess CLI. Off by default so the proven
    # subprocess path stays the default until validated on real GPU hardware.
    use_warm_predictor: bool = False
    # Lever 7: pin the resident predictor to a specific GPU / MIG slice (None = default).
    gpu_id: int | None = None
