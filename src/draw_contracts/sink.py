"""Trivial StatusSink implementations usable anywhere, including tests."""

from __future__ import annotations


class NullStatusSink:
    """A sink that records nothing. Satisfies the StatusSink Protocol.

    Useful for the CLI ``predict`` path (no queue) and for unit tests that do not
    care about status side effects.
    """

    def record_predicted(self, series_name: str, output_path: str) -> None:
        return None

    def record_failed(self, job_id: str, error: str) -> None:
        return None
