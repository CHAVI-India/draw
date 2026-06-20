"""Apply a determined nnU-Net postprocessing pickle to a folder of predictions.

Ported from ``draw/postprocess/postprocess.py``. The adapter is now a required
argument — the legacy module defaulted to a global ``default_nnunet_adapter`` that
was constructed (and mutated env) at import time.
"""

from __future__ import annotations

import os

from draw_core.accessor.nnunetv2 import NNUNetV2Adapter


def postprocess_folder(
    input_folder: str,
    output_folder: str,
    pkl_file: str,
    adapter: NNUNetV2Adapter,
) -> None:
    os.makedirs(output_folder, exist_ok=True)
    adapter.apply_postprocessing(input_folder, output_folder, pkl_file)
