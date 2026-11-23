"""Metal multi-stream concurrency and stream synchronization."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional

_tls = threading.local()


class MetalStream:
    """An independent execution stream wrapping a Metal command queue."""

    def __init__(
        self, stream_id: int = 0, priority: int = 0, *, _default: bool = False
    ) -> None:
        self.stream_id = stream_id
        self.priority = priority
        self._default = _default
        self._queue: Optional[Any] = None
        self._lock = threading.Lock()
        if not _default and self.available():
            self._queue = self._create_queue()

    @property
    def queue(self) -> Optional[Any]:
        if self._default:
            import parallax.metal_engine as me

            return me._COMMAND_QUEUE if self.available() else None
        return self._queue

    def available(self) -> bool:
        import parallax.metal_engine as me

        return bool(me.is_metal_available())

    def _create_queue(self) -> Optional[Any]:
        import ctypes
        import parallax.metal_engine as me

        if me._DEVICE is None or me._objc_send is None:
            return None
        return me._objc_send(ctypes.c_void_p, me._DEVICE, "newCommandQueue")

    def synchronize(self) -> None:
        queue = self.queue
        if not self.available() or queue is None:
            return
        import ctypes
        import parallax.metal_engine as me

        if me._objc_send is None:
            return
        cmd_buf = me._objc_send(ctypes.c_void_p, queue, "commandBuffer")
        if cmd_buf:
            me._objc_send(None, cmd_buf, "commit")
            me._objc_send(None, cmd_buf, "waitUntilCompleted")

    def release(self) -> None:
        if self._default:
            return
        with self._lock:
            if self._queue is not None:
                import parallax.metal_engine as me

                try:
                    if me._objc_send is not None:
                        me._objc_send(None, self._queue, "release")
                except Exception:
                    pass
                self._queue = None


_default_stream = MetalStream(stream_id=0, _default=True)


def current_stream() -> MetalStream:
    return getattr(_tls, "stream", _default_stream)


@contextmanager
def use_stream(stream: MetalStream) -> Generator[MetalStream, None, None]:
    if not isinstance(stream, MetalStream):
        raise TypeError(f"Expected MetalStream, got {type(stream).__name__}")
    prev = getattr(_tls, "stream", None)
    _tls.stream = stream
    try:
        yield stream
    finally:
        if prev is None:
            if hasattr(_tls, "stream"):
                delattr(_tls, "stream")
        else:
            _tls.stream = prev
