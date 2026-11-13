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

    @staticmethod
    def _bucket_size(size: int) -> int:
        if size <= 256:
            return 256
        return 1 << (size - 1).bit_length()

    def _allocate_raw(self, size: int) -> Optional[Any]:
        import parallax.metal_engine as me

        if me._DEVICE is None or me._objc_send is None:
            return None
        buf = me._objc_send(
            me.ctypes.c_void_p, me._DEVICE, "newBufferWithLength:options:", size, 0
        )
        if buf:
            self._allocated_bytes += size
        return buf

    def _free_raw(self, buffer: Any, size: int) -> None:
        import parallax.metal_engine as me

        try:
            if me._objc_send is not None:
                me._objc_send(None, buffer, "release")
        except Exception:
            pass
        self._allocated_bytes = max(0, self._allocated_bytes - size)
