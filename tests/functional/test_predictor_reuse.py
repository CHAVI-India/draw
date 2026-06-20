"""segment_study reuses ONE predictor across all submodels (GPU perf lever 2).

The big no-retrain GPU win is loading model weights once and reusing the resident
predictor for every submodel of a study, instead of cold-starting per call. We don't
need a GPU to verify the *wiring* that enables it: inject a fake Predictor and assert
the same instance is asked to predict each submodel (so a warm predictor would load
each model at most once).
"""

from __future__ import annotations

import os
import shutil

import nibabel as nib
import numpy as np

from draw_core.config import CoreConfig
from draw_core.models import ModelConfig, SubModel
from draw_core.segment import segment_study


class FakeAdapter:
    """Only needs apply_postprocessing here (no submodel uses it, but kept for parity)."""

    def apply_postprocessing(self, input_folder, output_folder, pkl_file):
        os.makedirs(output_folder, exist_ok=True)
        for f in os.listdir(input_folder):
            shutil.copy(os.path.join(input_folder, f), os.path.join(output_folder, f))


class RecordingPredictor:
    """Fake Predictor: records which submodels it served and drops a canned NIfTI."""

    def __init__(self, prediction_array, affine):
        self._array = prediction_array
        self._affine = affine
        self.served_dataset_ids: list[int] = []

    def predict_submodel(self, submodel, samples_dir, output_dir):
        self.served_dataset_ids.append(submodel.dataset_id)
        os.makedirs(output_dir, exist_ok=True)
        nib.save(
            nib.Nifti1Image(self._array, self._affine),
            os.path.join(output_dir, "seg_000.nii.gz"),
        )


class RecordingSink:
    def __init__(self):
        self.predicted = []

    def record_predicted(self, series_name, output_path):
        self.predicted.append((series_name, output_path))

    def record_failed(self, job_id, error):
        pass


def _two_submodel_model(seg_map):
    return ModelConfig(
        name="TSMulti",
        protocol="multi",
        submodels={
            810: SubModel(810, "TSMultiA", "3d_fullres", "nnUNetTrainer", None, seg_map),
            811: SubModel(811, "TSMultiB", "3d_fullres", "nnUNetTrainer", None, seg_map),
        },
    )


def test_single_predictor_instance_serves_every_submodel(tmp_path, ct_series_dir, multilabel_nifti):
    ct_dir, _series_uid, _ = ct_series_dir
    nifti_path, seg_map = multilabel_nifti
    pred = np.asanyarray(nib.load(nifti_path).dataobj).astype(np.uint8)

    predictor = RecordingPredictor(pred, np.eye(4))
    config = CoreConfig(nnunet_raw_dir=str(tmp_path / "raw"))

    result = segment_study(
        dicom_dirs=[ct_dir],
        preds_dir=str(tmp_path / "preds"),
        model=_two_submodel_model(seg_map),
        adapter=FakeAdapter(),
        config=config,
        result_sink=RecordingSink(),
        only_original=True,
        predictor=predictor,  # inject: one instance for the whole study
    )

    # The one predictor served BOTH submodels — i.e. a warm predictor would have
    # loaded each model once and reused them, never cold-starting per call.
    assert predictor.served_dataset_ids == [810, 811]
    assert len(result.series) == 2


def test_default_predictor_is_subprocess_when_warm_disabled():
    from draw_core.accessor.predictor import SubprocessPredictor
    from draw_core.logging import get_logger
    from draw_core.segment import _build_predictor

    predictor = _build_predictor(adapter=object(), config=CoreConfig(), log=get_logger("t"))

    assert isinstance(predictor, SubprocessPredictor)
