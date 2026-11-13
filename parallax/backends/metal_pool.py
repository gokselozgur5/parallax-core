"""Metal buffer pool and zero-copy view cache."""

from __future__ import annotations

import sys
import threading
from typing import Any, Optional
import numpy as np
from parallax.metal_engine import is_metal_available


class MetalBufferPool:
    """Thread-safe reusable Metal buffer pool."""

    def __init__(self, max_cached_bytes: int = 512 * 1024 * 1024) -> None:
        self.max_cached_bytes = max_cached_bytes
        self._lock = threading.Lock()
        self._pool: dict[int, list[Any]] = {}
        self._allocated_bytes = 0
        self._cached_bytes = 0
        self._hits = 0
        self._misses = 0

    def available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return is_metal_available()
