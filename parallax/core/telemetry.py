"""Dispatch telemetry counters and performance metrics."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class OpStats:
    """Aggregated invocation statistics for an operator."""

    total_count: int = 0
    backend_counts: dict[str, int] = field(default_factory=dict)
    total_latency_ns: int = 0


class DispatchTelemetry:
    """Thread-safe telemetry collector tracking op counts and latency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._op_stats: dict[str, OpStats] = {}
        self._total_ops: int = 0

    def total_ops(self) -> int:
        """Return total operations dispatched across all ops."""
        with self._lock:
            return self._total_ops

    def op_count(self, op: str) -> int:
        """Return total calls recorded for a specific op."""
        with self._lock:
            stats = self._op_stats.get(op)
            return stats.total_count if stats is not None else 0
