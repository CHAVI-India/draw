"""The engine contract: the ONLY thing the pipeline knows about a model.

DRAW's pipeline guarantees one thing: a DICOM series goes in, a DICOM RT-Struct
comes out. *How* a labeled mask is produced — nnU-Net, nnFormer, a promptable/LLM
model, a remote API — is entirely the engine's concern. The pipeline depends only on
this Protocol, so models are swappable without touching the queue, watcher,
conversion, or deployment layers.

Design choices that keep this general (not nnU-Net-shaped):

* ``segment`` takes the **label map in** (``seg_map``: id -> structure name). For
  nnU-Net it's metadata; for a promptable/LLM model the names ARE the prompts.
* ``segment`` returns **per-label masks** (``LabeledMask``), the most general shape:
  a promptable model emits one binary mask per structure (possibly overlapping); a
  multilabel model just splits its volume. The shared conversion layer combines them.
* No NIfTI / folder / patch / GPU / fold / checkpoint assumptions appear here. Those
  are nnU-Net implementation details and live inside ``NnUNetEngine``.

Training is deliberately NOT on this Protocol (see ``TrainableEngine``): a remote or
import-only engine has no training step and must not be forced to fake one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class LabeledMask:
    """One predicted structure for one study.

    ``mask`` is a boolean/binary 3D array in the source CT's voxel grid. ``name`` is
    the clinical structure name (what becomes an RT-Struct ROI). ``series_uid`` ties
    the mask back to its source DICOM series so the shared layer can locate the
    reference images. Masks for the same study/series are combined downstream into a
    single multi-label RT-Struct.
    """

    series_uid: str
    name: str
    mask: np.ndarray


@runtime_checkable
class SegmentationEngine(Protocol):
    """Turns DICOM studies into labeled masks. The pluggable unit.

    Implementations are obtained from the engine factory and are free to do any
    preprocessing/inference/postprocessing internally — the pipeline neither knows
    nor cares.
    """

    #: Stable identifier used as the config ``engine`` discriminator (e.g. "nnunet").
    name: str

    def segment(
        self,
        study_dirs: list[str],
        seg_map: dict[int, str],
        work_dir: str,
    ) -> list[LabeledMask]:
        """Segment one or more DICOM study directories.

        Args:
            study_dirs: local directories, each holding one DICOM series.
            seg_map: label id -> structure name. Metadata for nnU-Net; prompt source
                for promptable/LLM engines.
            work_dir: a scratch directory the engine may use freely.

        Returns:
            All produced structures across the given studies, as ``LabeledMask``
            objects keyed by ``series_uid``.
        """
        ...


@runtime_checkable
class TrainableEngine(SegmentationEngine, Protocol):
    """Optional capability: engines that can be trained/planned on prepared data.

    nnU-Net implements this; ONNX/remote/LLM engines do not. The training CLI checks
    ``isinstance(engine, TrainableEngine)`` and fails clearly otherwise, so no engine
    is forced to provide a meaningless ``train``.
    """

    def plan(self, dataset_id: str, **kwargs) -> None: ...

    def train(self, dataset_id: str, fold: str, **kwargs) -> None: ...
