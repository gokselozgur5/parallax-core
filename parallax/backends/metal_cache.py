"""Metal pipeline compilation cache."""

from __future__ import annotations

import hashlib
import sys
import threading
from typing import Any, Optional

from parallax.metal_engine import is_metal_available


class MetalPipelineCache:
    """Thread-safe cache for compiled Metal compute pipeline states."""

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], Any] = {}
        self._hits = 0
        self._misses = 0

    def available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return is_metal_available()
