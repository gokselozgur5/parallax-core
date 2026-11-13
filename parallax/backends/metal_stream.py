"""Metal multi-stream concurrency and stream synchronization."""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional

from parallax.metal_engine import is_metal_available

_tls = threading.local()


class MetalStream:
    """An independent execution stream wrapping a Metal command queue."""

    def __init__(self, stream_id: int = 0, priority: int = 0) -> None:
        self.stream_id = stream_id
        self.priority = priority
        self._queue: Optional[Any] = None
        self._lock = threading.Lock()
        if self.available():
            self._queue = self._create_queue()

    def available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return is_metal_available()

    def _create_queue(self) -> Optional[Any]:
        import ctypes
        import parallax.metal_engine as me

        if me._DEVICE is None or me._objc_send is None:
            return None
        return me._objc_send(ctypes.c_void_p, me._DEVICE, "newCommandQueue")

    def synchronize(self) -> None:
        if not self.available() or self._queue is None:
            return
        import ctypes
        import parallax.metal_engine as me

        cmd_buf = me._objc_send(ctypes.c_void_p, self._queue, "commandBuffer")
        if cmd_buf:
            me._objc_send(None, cmd_buf, "commit")
            me._objc_send(None, cmd_buf, "waitUntilCompleted")
