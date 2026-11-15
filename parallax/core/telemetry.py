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
