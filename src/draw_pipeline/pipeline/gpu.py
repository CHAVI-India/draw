"""GPU memory probe (infrastructure).

Moved out of ``draw.utils.ioutils`` into the scaffolding layer — it shells out to
``nvidia-smi`` and is pure infra, not segmentation policy. Made robust: csv with no
header/units, and a ``FileNotFoundError`` fallback (no nvidia-smi -> report 0 free).
"""

from __future__ import annotations

import subprocess

from draw_core.logging import get_logger

log = get_logger(__name__)


def get_gpu_memory() -> int:
    """Return free memory (MB) of the first GPU, or 0 if nvidia-smi is unavailable."""
    cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.check_output(cmd).decode("ascii").strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.warning("nvidia-smi unavailable; reporting 0 MB free GPU", exc_info=True)
        return 0
    if not out:
        return 0
    return int(out[0].split()[0])
