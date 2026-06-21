"""Engine factory: resolve the ``engine`` discriminator to a built engine.

Engines register a *builder* (a zero-heavy-import callable) under a name. The builder
is only invoked when that engine is actually requested, so importing this module — or
the whole ``draw_core`` package — never imports torch/nnU-Net/MONAI. This mirrors the
lazy-import discipline the rest of the core already follows.

Config flows as Option A: a generic envelope carries an ``engine`` name and an opaque
``engine_config`` dict; the factory hands that dict to the engine's builder, which
parses/validates it into the engine's own typed config (``NnUNetConfig`` etc.).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from draw_core.engines.base import SegmentationEngine
from draw_core.logging import get_logger

log = get_logger(__name__)

# engine name -> builder(engine_config: dict, **ctx) -> SegmentationEngine
_BUILDERS: dict[str, Callable[..., SegmentationEngine]] = {}

DEFAULT_ENGINE = "nnunet"


def register_engine(name: str) -> Callable[[Callable[..., SegmentationEngine]],
                                           Callable[..., SegmentationEngine]]:
    """Decorator to register an engine builder under ``name``.

    The builder receives the opaque ``engine_config`` dict (plus any context kwargs
    the call site passes, e.g. ``core_config``) and returns a ready engine.
    """

    def deco(builder: Callable[..., SegmentationEngine]) -> Callable[..., SegmentationEngine]:
        _BUILDERS[name] = builder
        return builder

    return deco


def _ensure_builtin_engines_imported() -> None:
    """Import built-in engine modules so their @register_engine runs. Lazy: only the
    requested engine's heavy deps load, because each module defers torch/nnU-Net
    imports to inside its build/segment path."""
    # nnU-Net is the default and always available to register (its heavy imports are
    # themselves deferred inside the engine).
    import draw_core.engines.nnunet  # noqa: F401


def available_engines() -> list[str]:
    _ensure_builtin_engines_imported()
    return sorted(_BUILDERS)


def build_engine(
    engine_name: str | None,
    engine_config: dict[str, Any] | None = None,
    **ctx: Any,
) -> SegmentationEngine:
    """Build the engine named ``engine_name`` (default nnU-Net) from its config block.

    Args:
        engine_name: the config ``engine`` discriminator. ``None`` -> nnU-Net, so
            existing configs with no ``engine`` key keep working.
        engine_config: the opaque per-engine config block (validated by the engine).
        **ctx: extra context (e.g. ``core_config``) passed through to the builder.
    """
    name = engine_name or DEFAULT_ENGINE
    _ensure_builtin_engines_imported()
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown engine {name!r}. Available: {sorted(_BUILDERS)}. "
            f"Is its optional extra installed (e.g. draw[{name}])?"
        )
    log.info("Building engine %r", name)
    return builder(engine_config or {}, **ctx)
