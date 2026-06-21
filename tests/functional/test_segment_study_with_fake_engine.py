"""segment_study is engine-agnostic: a fake engine plugs in with zero core changes.

This is the headline pluggability test. A FakeEngine that has nothing to do with
nnU-Net (no torch, no GPU, no NIfTI folders) produces LabeledMasks; segment_study
groups them per study and writes RT-Structs. Proves the contract: DICOM in, RT-Struct
out, engine internals irrelevant.
"""

from __future__ import annotations

import os

import numpy as np
from rt_utils import RTStructBuilder

from draw_conversion.dicom_io import get_series_instance_uid
from draw_core.config import CoreConfig
from draw_core.engines.base import LabeledMask
from draw_core.models import ModelConfig
from draw_core.segment import segment_study


class RecordingSink:
    def __init__(self):
        self.predicted = []

    def record_predicted(self, series_name, output_path):
        self.predicted.append((series_name, output_path))

    def record_failed(self, job_id, error):
        pass


class FakeEngine:
    """A stand-in model: emits canned per-structure masks. No nnU-Net, no GPU."""

    name = "fake"

    def __init__(self, shape):
        self.shape = shape
        self.segment_calls = 0
        self.received_seg_map = None

    def segment(self, study_dirs, seg_map, work_dir):
        self.segment_calls += 1
        self.received_seg_map = seg_map
        out = []
        for d in study_dirs:
            uid = get_series_instance_uid(d)
            # Two structures per study, proving multi-label combine in one RT-Struct.
            m1 = np.zeros(self.shape, dtype=bool)
            m1[8:16, 8:16, 1:5] = True
            m2 = np.zeros(self.shape, dtype=bool)
            m2[20:28, 20:28, 1:5] = True
            out.append(LabeledMask(series_uid=uid, name="Bladder", mask=m1))
            out.append(LabeledMask(series_uid=uid, name="CTVn", mask=m2))
        return out


def _fake_model():
    return ModelConfig(
        name="TSFake", protocol="fake", engine="fake",
        engine_config={}, label_map={1: "Bladder", 2: "CTVn"},
    )


def test_fake_engine_plugs_in_and_produces_rtstruct(tmp_path, ct_series_dir):
    ct_dir, _uid, (rows, cols, n_slices) = ct_series_dir
    # rt_utils expects (rows, cols, slices) after the nifti2rt transpose; the mask the
    # engine emits is in that oriented space already here.
    engine = FakeEngine((rows, cols, n_slices))
    sink = RecordingSink()

    result = segment_study(
        dicom_dirs=[ct_dir],
        preds_dir=str(tmp_path / "preds"),
        model=_fake_model(),
        config=CoreConfig(nnunet_raw_dir=str(tmp_path / "raw")),
        result_sink=sink,
        engine=engine,   # inject: no factory, no nnU-Net
    )

    assert engine.segment_calls == 1
    assert engine.received_seg_map == {1: "Bladder", 2: "CTVn"}  # label map passed IN
    assert len(result.series) == 1
    assert sink.predicted

    # Both structures landed in ONE RT-Struct for the study.
    rt_dir = result.series[0].output_path
    rt_path = os.path.join(rt_dir, [f for f in os.listdir(rt_dir) if f.endswith(".dcm")][0])
    names = RTStructBuilder.create_from(ct_dir, rt_path).get_roi_names()
    assert sorted(names) == ["Bladder", "CTVn"]
