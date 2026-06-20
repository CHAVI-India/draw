"""ModelRegistry parses the real config_yaml into typed dataclasses."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from draw_core.models import ModelRegistry

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config_yaml"


@pytest.fixture
def ts_prime_dir(tmp_path):
    """A registry dir with only ts_prime.yml (template.yml shares the prostate
    protocol, so isolate it for a deterministic protocol->name assertion)."""
    shutil.copy(CONFIG_DIR / "ts_prime.yml", tmp_path / "ts_prime.yml")
    return str(tmp_path)


def test_parses_ts_prime(ts_prime_dir):
    registry = ModelRegistry.from_yaml_dir(ts_prime_dir)
    model = registry.get("TSPrime")

    assert model.name == "TSPrime"
    assert model.protocol == "prostate"
    assert set(model.submodels.keys()) == {720, 721, 722}

    sub720 = model.submodels[720]
    assert sub720.dataset_id == 720
    assert sub720.config == "3d_fullres"
    assert sub720.trainer_name == "nnUNetTrainerNoMirroring"
    assert len(sub720.seg_map) == 6
    assert sub720.seg_map[1] == "Bladder"


def test_protocol_to_model_name(ts_prime_dir):
    registry = ModelRegistry.from_yaml_dir(ts_prime_dir)
    assert registry.protocol_to_model_name("prostate") == "TSPrime"
    assert registry.protocol_to_model_name("nonexistent") is None


def test_names_includes_tsprime(ts_prime_dir):
    registry = ModelRegistry.from_yaml_dir(ts_prime_dir)
    assert "TSPrime" in registry.names()
