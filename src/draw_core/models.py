"""Typed model registry parsed from the ``config_yaml/*.yml`` model definitions.

Replaces the legacy ``ALL_SEG_MAP`` / ``PROTOCOL_TO_MODEL`` global dicts (built at
import time inside ``draw/config.py``) and the stringly-typed access patterns like
``task_map[dataset_id]["map"]``. The YAML structure is preserved exactly; here it
is validated with the same ``schema`` rules and parsed into dataclasses so callers
work with typed objects (``SubModel.seg_map`` instead of ``model["map"]``).

This module is pure: it reads YAML files passed to it but performs no env reads and
has no import-time side effects.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import yaml
from schema import And, Or, Schema, SchemaError

from draw_core.logging import get_logger

log = get_logger(__name__)

# Same validation rules as the legacy draw/utils/mapping.py, kept for compatibility.
_MODEL_SCHEMA = Schema(
    {
        "name": str,
        "config": str,
        "map": {And(int, lambda n: n > 0): str},
        "trainer_name": str,
        "postprocess": Or(str, None),
    }
)

_CONF_SCHEMA = Schema(
    {
        "name": str,
        "protocol": str,
        # 0-10 reserved for MSD. Avoid.
        "models": {And(int, lambda n: n > 10): _MODEL_SCHEMA},
    }
)


@dataclass(frozen=True)
class SubModel:
    """One nnU-Net dataset within a site model (e.g. dataset 720 of TSPrime)."""

    dataset_id: int
    name: str
    config: str
    trainer_name: str
    postprocess: str | None
    seg_map: dict[int, str]


@dataclass(frozen=True)
class ModelConfig:
    """A per-site model: a protocol plus its ordered submodels keyed by dataset id."""

    name: str
    protocol: str
    submodels: dict[int, SubModel] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRegistry:
    """All parsed ``ModelConfig`` entries keyed by model name (e.g. ``"TSPrime"``)."""

    configs: dict[str, ModelConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml_dir(cls, config_root_dir: str) -> ModelRegistry:
        """Parse every ``*.yml`` under ``config_root_dir`` into the registry.

        Files that fail schema validation are skipped with a warning (matching the
        legacy behaviour), so a malformed template file does not break startup.
        """
        configs: dict[str, ModelConfig] = {}
        for file_name in glob.glob(
            os.path.join(config_root_dir, "**", "*.yml"), recursive=True
        ):
            model = cls._parse_file(file_name)
            if model is None:
                log.warning("Skipped %s due to schema problems", file_name)
                continue
            configs[model.name] = model
        return cls(configs=configs)

    @staticmethod
    def _parse_file(file_name: str) -> ModelConfig | None:
        with open(file_name) as stream:
            try:
                raw = yaml.safe_load(stream)
            except yaml.YAMLError:
                log.warning("Exception while reading YAML %s", file_name, exc_info=True)
                return None
        try:
            _CONF_SCHEMA.validate(raw)
        except SchemaError:
            log.error("Error while validating schema for %s", file_name, exc_info=True)
            return None

        submodels = {
            dataset_id: SubModel(
                dataset_id=dataset_id,
                name=spec["name"],
                config=spec["config"],
                trainer_name=spec["trainer_name"],
                postprocess=spec["postprocess"],
                seg_map=dict(spec["map"]),
            )
            for dataset_id, spec in raw["models"].items()
        }
        return ModelConfig(name=raw["name"], protocol=raw["protocol"], submodels=submodels)

    def get(self, name: str) -> ModelConfig:
        return self.configs[name]

    def names(self) -> list[str]:
        return list(self.configs.keys())

    def protocol_to_model_name(self, protocol: str) -> str | None:
        for cfg in self.configs.values():
            if cfg.protocol == protocol:
                return cfg.name
        return None
