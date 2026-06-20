"""Export a trained model dataset folder to a ZIP archive.

Ported from ``draw/impex/export.py``. Bug fix: the legacy lookup
``ALL_SEG_MAP[model]["models"][id]["name"]`` had an extra ``["models"]`` level that
did not match the actual structure. Here the dataset name comes from the typed
``ModelConfig.submodels[dataset_id].name``.
"""

from __future__ import annotations

import os
import zipfile

from draw_core.config import CoreConfig
from draw_core.logging import get_logger
from draw_core.models import ModelConfig

log = get_logger(__name__)


def export_to_zip(dataset_id: int, model: ModelConfig, config: CoreConfig) -> None:
    """Create a ZIP archive of the trained dataset's results folder."""
    dataset_name = model.submodels[int(dataset_id)].name
    source_dir = f"{config.nnunet_results_dir}/Dataset{dataset_id}_{dataset_name}"

    if not os.path.exists(source_dir):
        log.error("Source directory '%s' not found.", source_dir)
        return

    zip_file_path = f"Dataset{dataset_id}_{dataset_name}.zip"
    with zipfile.ZipFile(zip_file_path, "w") as zipf:
        for root, _dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, source_dir))

    log.info("ZIP file created successfully: %s", zip_file_path)
