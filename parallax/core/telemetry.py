"""Dispatch telemetry counters and performance metrics."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from parallax.backends.telemetry import DispatchTelemetry as BackendTelemetry


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

    def record(self, op: str, backend: str, latency_ns: int = 0) -> None:
        """Record an operation dispatch invocation with its latency."""
        if latency_ns < 0:
            raise ValueError(f"latency_ns must be non-negative, got {latency_ns}")
        with self._lock:
            self._total_ops += 1
            stats = self._op_stats.get(op)
            if stats is None:
                stats = OpStats()
                self._op_stats[op] = stats
            stats.total_count += 1
            stats.backend_counts[backend] = stats.backend_counts.get(backend, 0) + 1
            stats.total_latency_ns += latency_ns
        BackendTelemetry().record(op, backend, latency_ns)

    def backend_count(self, backend: str, op: str | None = None) -> int:
        """Return total calls executed on backend, optionally filtered by op."""
        with self._lock:
            if op is not None:
                stats = self._op_stats.get(op)
                return stats.backend_counts.get(backend, 0) if stats else 0
            return sum(s.backend_counts.get(backend, 0) for s in self._op_stats.values())

    def backend_hit_rate(self, backend: str, op: str | None = None) -> float:
        """Return hit rate for backend, optionally filtered by op."""
        with self._lock:
            if op is not None:
                stats = self._op_stats.get(op)
                if stats is None or stats.total_count == 0:
                    return 0.0
                return stats.backend_counts.get(backend, 0) / stats.total_count
            if self._total_ops == 0:
                return 0.0
            hits = sum(s.backend_counts.get(backend, 0) for s in self._op_stats.values())
            return hits / self._total_ops

    def total_latency_ns(self, op: str | None = None) -> int:
        """Return total elapsed nanoseconds, optionally filtered by op."""
        with self._lock:
            if op is not None:
                stats = self._op_stats.get(op)
                return stats.total_latency_ns if stats is not None else 0
            return sum(s.total_latency_ns for s in self._op_stats.values())

    def average_latency_ns(self, op: str | None = None) -> float:
        """Return mean latency in nanoseconds, optionally filtered by op."""
        with self._lock:
            if op is not None:
                stats = self._op_stats.get(op)
                if stats is None or stats.total_count == 0:
                    return 0.0
                return stats.total_latency_ns / stats.total_count
            if self._total_ops == 0:
                return 0.0
            total_ns = sum(s.total_latency_ns for s in self._op_stats.values())
            return total_ns / self._total_ops

    def reset(self) -> None:
        """Reset all tracked counters and statistics to zero."""
        with self._lock:
            self._op_stats.clear()
            self._total_ops = 0
        BackendTelemetry().reset()


_DEFAULT_TELEMETRY = DispatchTelemetry()


def get_telemetry() -> DispatchTelemetry:
    """Return the global default DispatchTelemetry instance."""
    return _DEFAULT_TELEMETRY


def record(op: str, backend: str, latency_ns: int = 0) -> None:
    """Record an operation dispatch invocation on default telemetry."""
    _DEFAULT_TELEMETRY.record(op, backend, latency_ns)


def get_op_count(op: str) -> int:
    """Return total count for an op on default telemetry."""
    return _DEFAULT_TELEMETRY.op_count(op)


def get_backend_hit_rate(backend: str, op: str | None = None) -> float:
    """Return backend hit rate on default telemetry."""
    return _DEFAULT_TELEMETRY.backend_hit_rate(backend, op=op)


def get_average_latency_ns(op: str | None = None) -> float:
    """Return average latency on default telemetry."""
    return _DEFAULT_TELEMETRY.average_latency_ns(op=op)


def reset_telemetry() -> None:
    """Reset default telemetry counters to zero."""
    _DEFAULT_TELEMETRY.reset()
