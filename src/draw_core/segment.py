"""High-level segmentation policy — engine-agnostic.

The pipeline's only contract: DICOM study dirs in, DICOM RT-Structs out. ``segment_study``
builds the configured engine from the factory, asks it for per-structure masks, and
writes one multi-label RT-Struct per study via the shared conversion layer. It contains
NO model-specific logic — nnU-Net, nnFormer, a promptable/remote model are all just an
``engine`` behind the same Protocol.

What stays here (generic): grouping masks by study, locating each study's reference
DICOM, writing the RT-Struct, recording status. What used to be here (nnU-Net dataset
folders, submodels, .pkl postproc, predictors) now lives in ``NnUNetEngine``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from draw_contracts.dto import SegmentationResult, SeriesResult
from draw_conversion.dicom_io import get_series_instance_uid
from draw_conversion.nifti2rt import write_named_masks_to_rtstruct
from draw_core.config import CoreConfig
from draw_core.engines.base import LabeledMask, SegmentationEngine
from draw_core.engines.factory import build_engine
from draw_core.logging import get_logger
from draw_core.models import ModelConfig

_default_log = get_logger(__name__)


def segment_study(
    dicom_dirs: list[str],
    preds_dir: str,
    model: ModelConfig,
    config: CoreConfig,
    result_sink,
    logger: logging.Logger | None = None,
    engine: SegmentationEngine | None = None,
) -> SegmentationResult:
    """Segment DICOM study dirs with ``model``'s engine and write RT-Structs.

    The engine (default nnU-Net, from ``model.engine`` + ``model.engine_config``)
    returns per-structure masks; we group them by source series and write one
    RT-Struct per study. ``engine`` may be injected (tests / a prebuilt engine).
    """
    log = logger or _default_log
    exp_number = datetime.now().strftime("%Y-%m-%d.%H-%M")
    final_output_dir = os.path.join(preds_dir, model.name, "results")
    work_dir = os.path.join(preds_dir, model.name)

    if engine is None:
        engine = build_engine(model.engine, model.engine_config, core_config=config)

    masks: list[LabeledMask] = engine.segment(dicom_dirs, model.label_map, work_dir)

    # Map each produced series uid back to its source DICOM dir (needed to build the
    # RT-Struct against the reference images).
    uid_to_dicom = {}
    for d in dicom_dirs:
        uid = get_series_instance_uid(d)
        if uid:
            uid_to_dicom[uid] = d

    # Group all of a study's structures so they land in ONE multi-label RT-Struct.
    by_series: dict[str, list[tuple[str, object]]] = {}
    for m in masks:
        by_series.setdefault(m.series_uid, []).append((m.name, m.mask))

    all_series: list[SeriesResult] = []
    for series_uid, named_masks in by_series.items():
        dicom_dir = uid_to_dicom.get(series_uid)
        if dicom_dir is None:
            log.warning("No source DICOM dir for series %s; skipping", series_uid)
            continue
        save_dir = write_named_masks_to_rtstruct(
            named_masks, dicom_dir, f"{final_output_dir}/{exp_number}/{series_uid}"
        )
        result_sink.record_predicted(series_uid, save_dir)
        all_series.append(SeriesResult(series_name=series_uid, output_path=save_dir))

    log.info("Segmentation complete for %s (%d series)", model.name, len(all_series))
    return SegmentationResult(
        job_id=exp_number, output_local_dir=final_output_dir, series=all_series
    )
