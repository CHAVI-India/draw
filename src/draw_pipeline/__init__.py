"""DRAW pipeline: scaffolding / details layer.

This is the low-level outer layer (DB, filesystem, watcher, CLI). It may depend on
heavy infrastructure (SQLAlchemy, watchdog) and on ``draw_core`` / ``draw_contracts``
/ ``draw_conversion``. The core never depends on this package.
"""
