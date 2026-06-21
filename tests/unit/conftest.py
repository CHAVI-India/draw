"""Unit-test fixtures.

Keeps the engine factory's module-level ``_BUILDERS`` registry hermetic: tests that
register a throwaway engine (``@register_engine("dummy-test-engine")`` etc.) otherwise
leak that registration into every later test in the session, since the module is
imported once. We snapshot the registry before each test and restore it after, so a
test's registrations cannot be observed by another test (Software Eng. at Google,
Ch. 11: tests must not share state).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_engine_registry():
    from draw_core.engines import factory

    saved = dict(factory._BUILDERS)
    try:
        yield
    finally:
        factory._BUILDERS.clear()
        factory._BUILDERS.update(saved)
