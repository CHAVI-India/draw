"""Opt-in faster gzip for NIfTI I/O via python-isal.

Benchmarked ~2.3-2.9x faster compression than stdlib zlib on a 63 MB CT volume.
nibabel uses the stdlib ``gzip`` module under the hood; ``isal.igzip`` is a
drop-in replacement. We swap it in process-wide if available, and no-op otherwise
so the conversion layer still works without the optional dependency.
"""

from __future__ import annotations

from draw_core.logging import get_logger

log = get_logger(__name__)


def enable_fast_gzip() -> bool:
    """Install isal as nibabel's gzip backend. Returns True if enabled."""
    try:
        from isal import igzip
    except ImportError:
        log.debug("python-isal not installed; using stdlib gzip")
        return False

    try:
        import nibabel.openers as openers

        # nibabel reads/writes .gz through this handle; isal.igzip is API-compatible.
        openers.HAVE_INDEXED_GZIP = False
        openers.GzipFile = igzip.IGzipFile  # type: ignore[attr-defined]
        log.debug("Enabled python-isal gzip backend for nibabel")
        return True
    except Exception:  # pragma: no cover - defensive
        log.warning("Could not install isal gzip backend; using stdlib gzip")
        return False
