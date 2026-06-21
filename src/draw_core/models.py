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

# Legacy schema: a flat nnU-Net config with a `models` block and no `engine` key.
# Still accepted verbatim so existing config_yaml/*.yml parse unchanged.
_CONF_SCHEMA = Schema(
    {
        "name": str,
        "protocol": str,
        # 0-10 reserved for MSD. Avoid.
        "models": {And(int, lambda n: n > 10): _MODEL_SCHEMA},
    }
)

# New generic envelope (Option A): an explicit `engine` + an opaque `engine_config`
# block that the engine validates itself. `map` is the generic label map the shared
# conversion layer needs (id -> structure name).
_ENGINE_CONF_SCHEMA = Schema(
    {
        "name": str,
        "protocol": str,
        "engine": str,
        "map": {And(int, lambda n: n > 0): str},
        "engine_config": dict,  # opaque to the core; the engine parses/validates it
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
    """A per-site model — the generic envelope the pipeline understands.

    The pipeline reads only ``name``, ``protocol``, ``engine``, and ``label_map``, and
    passes ``engine_config`` opaquely to the engine factory. ``submodels`` is kept for
    backward compatibility (it mirrors the legacy nnU-Net ``models`` block) but the
    generic layer no longer depends on it.
    """

    name: str
    protocol: str
    engine: str = "nnunet"
    engine_config: dict = field(default_factory=dict)
    label_map: dict[int, str] = field(default_factory=dict)
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
        # New generic envelope: explicit `engine` + opaque `engine_config`.
        if isinstance(raw, dict) and "engine" in raw:
            try:
                _ENGINE_CONF_SCHEMA.validate(raw)
            except SchemaError:
                log.error("Error validating engine schema for %s", file_name, exc_info=True)
                return None
            return ModelConfig(
                name=raw["name"],
                protocol=raw["protocol"],
                engine=raw["engine"],
                engine_config=raw["engine_config"],
                label_map=dict(raw["map"]),
            )

        # Legacy flat nnU-Net config (no `engine` key): default engine=nnunet, carry the
        # whole dict as the engine_config block, and derive the generic label map as the
        # union of submodel maps. Behaviour is unchanged for existing files.
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
        # nnU-Net submodels reuse label ids (each starts at 1), so a naive union would
        # collide and drop structures. Re-key to a flat, collision-free id space so the
        # generic layer sees EVERY structure name. (nnU-Net itself ignores this map —
        # structures are baked into the trained weights and returned via LabeledMask.)
        label_map: dict[int, str] = {}
        next_id = 1
        for sub in submodels.values():
            for name in sub.seg_map.values():
                label_map[next_id] = name
                next_id += 1
        return ModelConfig(
            name=raw["name"],
            protocol=raw["protocol"],
            engine="nnunet",
            engine_config={"models": raw["models"]},
            label_map=label_map,
            submodels=submodels,
        )

    def get(self, name: str) -> ModelConfig:
        return self.configs[name]

    def names(self) -> list[str]:
        return list(self.configs.keys())

    def protocol_to_model_name(self, protocol: str) -> str | None:
        for cfg in self.configs.values():
            if cfg.protocol == protocol:
                return cfg.name
        return None
