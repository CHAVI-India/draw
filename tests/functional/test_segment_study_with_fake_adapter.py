"""End-to-end segment_study on CPU with a fake adapter (real preprocess+conversion).

Exercises the orchestration in ``draw_core.segment.segment_study`` without a GPU:

* real preprocess runs on the synthetic CT (only_original=True needs no RTStruct),
* a FAKE adapter drops a canned multilabel prediction NIfTI into the output dir,
* the real NIfTI->RTStruct conversion produces a series,
* a recording fake StatusSink confirms ``record_predicted`` was called.
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
    """Stand-in for NNUNetV2Adapter: writes a canned prediction, copies for postproc."""

    def __init__(self, prediction_array, affine):
        self._array = prediction_array
        self._affine = affine
        self.predict_calls = 0

    def predict_folder(self, samples_dir, output_dir, model_config, dataset_id, fold, trainer_name):
        self.predict_calls += 1
        os.makedirs(output_dir, exist_ok=True)
        # nnU-Net names predictions after the sample (seg_000.nii.gz).
        nib.save(
            nib.Nifti1Image(self._array, self._affine),
            os.path.join(output_dir, "seg_000.nii.gz"),
        )

    def apply_postprocessing(self, input_folder, output_folder, pkl_file):
        os.makedirs(output_folder, exist_ok=True)
        for f in os.listdir(input_folder):
            shutil.copy(os.path.join(input_folder, f), os.path.join(output_folder, f))


class RecordingSink:
    def __init__(self):
        self.predicted = []
        self.failed = []

    def record_predicted(self, series_name, output_path):
        self.predicted.append((series_name, output_path))

    def record_failed(self, job_id, error):
        self.failed.append((job_id, error))


def test_segment_study_end_to_end(tmp_path, ct_series_dir, multilabel_nifti):
    ct_dir, _series_uid, (rows, cols, n_slices) = ct_series_dir
    seg_map = {1: "Bladder", 2: "Anorectum"}

    # Canned prediction matching the CT geometry, in nnU-Net's (z, y, x) orientation
    # that SimpleITK-written images carry; reuse the multilabel fixture's content.
    nifti_path, _ = multilabel_nifti
    pred = np.asanyarray(nib.load(nifti_path).dataobj).astype(np.uint8)
    adapter = FakeAdapter(pred, np.eye(4))

    raw_dir = tmp_path / "raw"
    config = CoreConfig(nnunet_raw_dir=str(raw_dir))

    model = ModelConfig(
        name="TSFake",
        protocol="fake",
        submodels={
            999: SubModel(
                dataset_id=999,
                name="TSFake",
                config="3d_fullres",
                trainer_name="nnUNetTrainer",
                postprocess=None,
                seg_map=seg_map,
            )
        },
    )

    sink = RecordingSink()
    result = segment_study(
        dicom_dirs=[ct_dir],
        preds_dir=str(tmp_path / "preds"),
        model=model,
        adapter=adapter,
        config=config,
        result_sink=sink,
        only_original=True,
    )

    assert adapter.predict_calls == 1
    assert sink.predicted, "record_predicted should have been called at least once"
    assert result.series, "SegmentationResult.series must be non-empty"
    # The produced RT-Struct dir exists.
    assert os.path.isdir(result.series[0].output_path)
