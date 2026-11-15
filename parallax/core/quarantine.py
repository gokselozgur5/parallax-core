"""Backend quarantine after repeated runtime failures."""
from __future__ import annotations
from collections.abc import Sequence
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

    def filter_backends(self, op: str, backends: Sequence[str]) -> list[str]:
        """Return candidate backends excluding quarantined ones for op."""
        with self._lock:
            return [b for b in backends if (op, b) not in self._quarantined]


    def failure_count(self, op: str, backend: str) -> int:
        with self._lock:
            return self._failures.get((op, backend), 0)

    def record_success(self, op: str, backend: str) -> None:
        """Reset consecutive failure counter on success."""
        with self._lock:
            self._failures.pop((op, backend), None)

    def reset(self, op: str | None = None, backend: str | None = None) -> None:
        """Reset quarantine and failure counts selectively or globally."""
        with self._lock:
            if op is None and backend is None:
                self._failures.clear()
                self._quarantined.clear()
                return
            keys = [
                k for k in self._failures
                if (op is None or k[0] == op) and (backend is None or k[1] == backend)
            ]
            for k in keys:
                self._failures.pop(k, None)
                self._quarantined.discard(k)



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

