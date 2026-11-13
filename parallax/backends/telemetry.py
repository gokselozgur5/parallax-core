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
