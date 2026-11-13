"""Per (op, backend) dispatch call counters."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class CallStats:
    """Number of calls and their summed duration in nanoseconds."""

    count: int = 0
    total_ns: int = 0


class DispatchTelemetry:
    """Thread-safe call counters keyed by (op, backend)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[tuple[str, str], CallStats] = {}

    def record(self, op: str, backend: str, elapsed_ns: int) -> None:
        """Add one call of ``elapsed_ns`` nanoseconds."""
        if elapsed_ns < 0:
            raise ValueError(f"elapsed_ns must be non-negative, got {elapsed_ns}")
        key = (op, backend)
        with self._lock:
            old = self._stats.get(key, CallStats())
            self._stats[key] = CallStats(old.count + 1, old.total_ns + elapsed_ns)

    def snapshot(self) -> Mapping[tuple[str, str], CallStats]:
        """Return a read-only copy of the current counters."""
        with self._lock:
            return MappingProxyType(dict(self._stats))

    def reset(self) -> None:
        """Clear every counter."""
        with self._lock:
            self._stats.clear()
