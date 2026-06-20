"""Regression guard: core + conversion + contracts import with no env file, no GPU.

The legacy ``draw/config.py`` read ``env.draw.yml`` at import and ``draw/utils/logging.py``
configured a file handler at import, so importing almost anything crashed without
those files in the CWD. These tests run from a temp CWD with nothing present and
must still succeed — they encode the central invariant of the refactor.
"""

from __future__ import annotations

import importlib
import os

import pytest

PURE_MODULES = [
    "draw_contracts",
    "draw_contracts.dto",
    "draw_contracts.protocols",
    "draw_contracts.sink",
    "draw_core.constants",
    "draw_core.config",
    "draw_core.logging",
    "draw_conversion.dicom_io",
    "draw_conversion.nifti2rt",
    "draw_conversion.dcm2nii",
]


@pytest.mark.parametrize("module", PURE_MODULES)
def test_imports_from_empty_cwd(module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no env.draw.yml, no logs/, no data/ here
    importlib.import_module(module)
    assert not (tmp_path / "logs").exists(), "import must not create logs/ dir"
    assert not os.path.exists(tmp_path / "env.draw.yml")


def test_core_does_not_import_torch():
    """Importing core must not pull in torch (it's behind the optional gpu extra)."""
    import sys

    for mod in ["draw_core", "draw_core.constants", "draw_conversion.nifti2rt"]:
        importlib.import_module(mod)
    assert "torch" not in sys.modules
