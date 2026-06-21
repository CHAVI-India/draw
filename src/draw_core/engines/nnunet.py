"""nnU-Net segmentation engine — the reference implementation and default.

This is where ALL nnU-Net-specific concepts live, hidden behind the generic
``SegmentationEngine`` contract: the per-submodel split for overlapping labels, the
``Dataset<id>_<name>/imagesTr`` folder format, the ``3d_fullres`` config, the
trainer name, the ``.pkl`` postprocessing, the fold/checkpoint, the
subprocess/warm predictor runtime, and the ``nnUNet_*`` env vars.

The engine takes DICOM study dirs in and returns ``LabeledMask`` objects out; it
reads its own predicted NIfTIs and splits them into per-structure boolean masks, so
the generic conversion layer never sees a NIfTI or a dataset folder.

Heavy imports (torch via the predictor) stay deferred inside the call path, so
importing this module to register the engine costs nothing.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from draw_core.config import CoreConfig
from draw_core.engines.base import LabeledMask
from draw_core.engines.factory import register_engine
from draw_core.logging import get_logger

log = get_logger(__name__)

SAMPLE_NUMBER_ZFILL = 3


@dataclass(frozen=True)
class NnUNetSubModel:
    """One nnU-Net dataset within a site model (e.g. dataset 720 of TSPrime)."""

    dataset_id: int
    name: str
    config: str
    trainer_name: str
    postprocess: str | None
    seg_map: dict[int, str]


@dataclass(frozen=True)
class NnUNetConfig:
    """nnU-Net engine config, parsed from the opaque ``engine_config`` block.

    The legacy ``models`` block maps directly onto ``submodels`` — so an existing
    ``config_yaml/*.yml`` (which has no ``engine`` key) parses unchanged.
    """

    submodels: dict[int, NnUNetSubModel]
    fold: str = "0"
    checkpoint_name: str = "checkpoint_best.pth"
    use_warm_predictor: bool = False
    gpu_id: int | None = None

    @classmethod
    def from_block(cls, block: dict, *, core_config: CoreConfig | None = None) -> NnUNetConfig:
        """Parse the engine_config block. ``block`` is the legacy model dict (has a
        ``models`` key) for backward compatibility."""
        models = block.get("models", {})
        submodels = {
            int(dsid): NnUNetSubModel(
                dataset_id=int(dsid),
                name=spec["name"],
                config=spec["config"],
                trainer_name=spec["trainer_name"],
                postprocess=spec["postprocess"],
                seg_map=dict(spec["map"]),
            )
            for dsid, spec in models.items()
        }
        cc = core_config or CoreConfig()
        return cls(
            submodels=submodels,
            fold=block.get("fold", "0"),
            checkpoint_name=block.get("checkpoint_name", cc.checkpoint_name),
            use_warm_predictor=block.get("use_warm_predictor", cc.use_warm_predictor),
            gpu_id=block.get("gpu_id", cc.gpu_id),
        )


class NnUNetEngine:
    """SegmentationEngine + TrainableEngine backed by nnU-Net."""

    name = "nnunet"

    def __init__(self, config: NnUNetConfig, core_config: CoreConfig):
        self.config = config
        self.core_config = core_config

    # ---- SegmentationEngine -----------------------------------------------------
    def segment(
        self,
        study_dirs: list[str],
        seg_map: dict[int, str],
        work_dir: str,
    ) -> list[LabeledMask]:
        """Run every submodel over the studies; return per-structure masks.

        ``seg_map`` is ignored here — nnU-Net's structures are defined per-submodel in
        its own config (baked into the trained weights). It is part of the contract
        for promptable engines.
        """
        from draw_core.accessor.nnunetv2 import NNUNetV2Adapter
        from draw_core.preprocess.preprocess_data import convert_dicom_dir_to_nnunet_dataset

        adapter = NNUNetV2Adapter(self.core_config)
        predictor = self._build_predictor(adapter)

        results: list[LabeledMask] = []
        for submodel in self.config.submodels.values():
            pred_dir, dataset_dir = self._run_submodel(
                submodel, study_dirs, work_dir, predictor, adapter,
                convert_dicom_dir_to_nnunet_dataset,
            )
            results.extend(self._masks_from_predictions(submodel, pred_dir, dataset_dir))
        return results

    def _build_predictor(self, adapter):
        from draw_core.accessor.predictor import SubprocessPredictor

        if self.config.use_warm_predictor:
            from draw_core.accessor.warm_predictor import WarmPredictor

            log.info("nnU-Net: warm in-process predictor (GPU levers 1/2/5/8)")
            return WarmPredictor(self.core_config, gpu_id=self.config.gpu_id, fold=self.config.fold)
        return SubprocessPredictor(adapter, fold=self.config.fold)

    def _run_submodel(
        self, submodel, study_dirs, work_dir, predictor, adapter, convert_fn
    ) -> tuple[str, str]:
        dataset_id = submodel.dataset_id
        dataset_dir = os.path.normpath(
            f"{self.core_config.nnunet_raw_dir}/Dataset{dataset_id}_{submodel.name}"
        )
        _remove(dataset_dir)
        for idx, dicom_dir in enumerate(study_dirs):
            sample_number = str(idx).zfill(SAMPLE_NUMBER_ZFILL)
            dataset_dir = convert_fn(
                dicom_dir, dataset_id, submodel.name, sample_number, submodel.seg_map,
                raw_dir=self.core_config.nnunet_raw_dir, only_original=True, logger=log,
            )

        tr_images = os.path.join(dataset_dir, "imagesTr")
        model_pred_dir = os.path.join(work_dir, str(dataset_id), "modelpred")
        _remove(model_pred_dir)
        os.makedirs(model_pred_dir, exist_ok=True)
        predictor.predict_submodel(_PredictTarget(submodel), tr_images, model_pred_dir)

        if submodel.postprocess is not None:
            op_folder = os.path.join(work_dir, str(dataset_id), "postprocess")
            _remove(op_folder)
            os.makedirs(op_folder, exist_ok=True)
            pkl_dest = f"{op_folder}/postprocessing.pkl"
            shutil.copy(submodel.postprocess, pkl_dest)
            from draw_core.postprocess.postprocess import postprocess_folder

            postprocess_folder(model_pred_dir, op_folder, pkl_dest, adapter)
            model_pred_dir = op_folder
        return model_pred_dir, dataset_dir

    def _masks_from_predictions(self, submodel, pred_dir, dataset_dir) -> list[LabeledMask]:
        """Read each predicted multilabel NIfTI and split it into per-structure masks."""
        import glob

        from draw_conversion.nifti2rt import (
            get_dcm_root,
            get_sample_number_from_nifti_path,
            make_mask_from_rt,
        )

        out: list[LabeledMask] = []
        for nifti_path in glob.glob(f"{pred_dir}/**.nii.gz"):
            sample_no = get_sample_number_from_nifti_path(nifti_path, "seg")
            _dcm_root, series_uid = get_dcm_root(submodel.dataset_id, sample_no, dataset_dir)
            if not series_uid:
                continue
            np_mask = make_mask_from_rt(nifti_path)
            for idx, name in submodel.seg_map.items():
                out.append(LabeledMask(series_uid=series_uid, name=name, mask=(np_mask == idx)))
        return out

    # ---- TrainableEngine --------------------------------------------------------
    def plan(self, dataset_id: str, **kwargs) -> None:
        from draw_core.accessor.nnunetv2 import NNUNetV2Adapter

        sub = self._submodel(dataset_id)
        NNUNetV2Adapter(self.core_config).plan(
            dataset_id, config=sub.config, gpu_memory_gb=kwargs.get("gpu_memory_gb")
        )

    def train(self, dataset_id: str, fold: str, **kwargs) -> None:
        from draw_core.accessor.nnunetv2 import NNUNetV2Adapter

        sub = self._submodel(dataset_id)
        NNUNetV2Adapter(self.core_config).train(
            dataset_id, sub.config, fold, sub.trainer_name,
            resume=kwargs.get("resume", True), device_id=kwargs.get("device_id", 0),
        )

    def _submodel(self, dataset_id: str) -> NnUNetSubModel:
        return self.config.submodels[int(dataset_id)]


def _remove(path: str) -> None:
    if os.path.exists(path):
        log.info("Deleting %s", path)
        shutil.rmtree(path)


class _PredictTarget:
    """Adapt an NnUNetSubModel to the Predictor seam's expected attributes."""

    def __init__(self, submodel: NnUNetSubModel):
        self.dataset_id = submodel.dataset_id
        self.name = submodel.name
        self.config = submodel.config
        self.trainer_name = submodel.trainer_name


@register_engine("nnunet")
def _build_nnunet(engine_config: dict, *, core_config: CoreConfig | None = None, **_):
    cc = core_config or CoreConfig()
    return NnUNetEngine(NnUNetConfig.from_block(engine_config, core_config=cc), cc)
