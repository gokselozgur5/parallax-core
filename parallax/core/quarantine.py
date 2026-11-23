"""Backend quarantine after repeated runtime failures."""
from __future__ import annotations
from collections.abc import Sequence
import threading
import time

class BackendQuarantine:
    """Tracks consecutive op failures and isolates failing backends."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 0.0) -> None:
        if failure_threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {failure_threshold}")
        self._threshold = failure_threshold
        self._cooldown = max(0.0, float(cooldown_seconds))
        self._lock = threading.Lock()
        self._failures: dict[tuple[str, str], int] = {}
        self._quarantined: set[tuple[str, str]] = set()
        self._quarantine_times: dict[tuple[str, str], float] = {}

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def cooldown(self) -> float:
        return self._cooldown

    def is_quarantined(self, op: str, backend: str) -> bool:
        with self._lock:
            key = (op, backend)
            if key not in self._quarantined:
                return False
            if self._cooldown > 0 and (time.monotonic() - self._quarantine_times.get(key, 0.0)) >= self._cooldown:
                self._quarantined.discard(key)
                self._failures.pop(key, None)
                return False
            return True


    def filter_backends(self, op: str, backends: Sequence[str]) -> list[str]:
        """Return candidate backends excluding quarantined ones for op."""
        return [b for b in backends if not self.is_quarantined(op, b)]



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
                self._quarantine_times[key] = time.monotonic()
                return True
            return False


_DEFAULT_QUARANTINE = BackendQuarantine()


def get_default_quarantine() -> BackendQuarantine:
    """Return the global default BackendQuarantine instance."""
    return _DEFAULT_QUARANTINE


def record_failure(op: str, backend: str) -> bool:
    """Record an op failure on the default quarantine."""
    return _DEFAULT_QUARANTINE.record_failure(op, backend)


def record_success(op: str, backend: str) -> None:
    """Record an op success on the default quarantine."""
    _DEFAULT_QUARANTINE.record_success(op, backend)


def is_quarantined(op: str, backend: str) -> bool:
    """Return True if (op, backend) is quarantined on default instance."""
    return _DEFAULT_QUARANTINE.is_quarantined(op, backend)


def filter_backends(op: str, backends: Sequence[str]) -> list[str]:
    """Return non-quarantined backends from default quarantine."""
    return _DEFAULT_QUARANTINE.filter_backends(op, backends)


def reset_quarantine(op: str | None = None, backend: str | None = None) -> None:
    """Reset default quarantine counters."""
    _DEFAULT_QUARANTINE.reset(op=op, backend=backend)


__all__ = [
    "BackendQuarantine",
    "filter_backends",
    "get_default_quarantine",
    "is_quarantined",
    "record_failure",
    "record_success",
    "reset_quarantine",
]




