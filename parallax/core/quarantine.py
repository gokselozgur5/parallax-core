"""Backend quarantine after repeated runtime failures."""
from __future__ import annotations
import threading

class BackendQuarantine:
    """Tracks consecutive op failures and isolates failing backends."""

    def __init__(self, failure_threshold: int = 3) -> None:
        if failure_threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {failure_threshold}")
        self._threshold = failure_threshold
        self._lock = threading.Lock()
        self._failures: dict[tuple[str, str], int] = {}
        self._quarantined: set[tuple[str, str]] = set()

    @property
    def threshold(self) -> int:
        return self._threshold

    def is_quarantined(self, op: str, backend: str) -> bool:
        with self._lock:
            return (op, backend) in self._quarantined

    def failure_count(self, op: str, backend: str) -> int:
        with self._lock:
            return self._failures.get((op, backend), 0)
