from __future__ import annotations

import threading
from typing import Optional


class BackendQuarantine:
    """Tracks consecutive op failures and quarantines failing backends."""

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

    def record_failure(self, op: str, backend: str) -> bool:
        """Record failure; returns True if pair is now quarantined."""
        with self._lock:
            key = (op, backend)
            count = self._failures.get(key, 0) + 1
            self._failures[key] = count
            if count >= self._threshold:
                self._quarantined.add(key)
                return True
            return False

    def record_success(self, op: str, backend: str) -> None:
        """Reset consecutive failure counter on success."""
        with self._lock:
            self._failures.pop((op, backend), None)

    def is_quarantined(self, op: str, backend: str) -> bool:
        """Return True if (op, backend) is currently quarantined."""
        with self._lock:
            return (op, backend) in self._quarantined

    def failure_count(self, op: str, backend: str) -> int:
        """Return current consecutive failure count for (op, backend)."""
        with self._lock:
            return self._failures.get((op, backend), 0)
