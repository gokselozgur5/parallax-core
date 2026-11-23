from __future__ import annotations

import ctypes
import types
from typing import Any
import numpy as np
import pytest
import parallax.metal_engine as me
from parallax.backends.metal_stream import MetalStream, use_stream


def _install_queue_spy(monkeypatch: pytest.MonkeyPatch, force_stub: bool = False) -> list[Any]:
    queues: list[Any] = []
    if not force_stub and me.is_metal_available():
        real_send = me._objc_send
        monkeypatch.setattr(me, "_objc_send", lambda r, t, s, *a: (queues.append(t) if s == "commandBuffer" else None, real_send(r, t, s, *a))[1])
        return queues
    buf, seq = (ctypes.c_float * 64)(), [30]
    cb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, me.MTLSize, me.MTLSize)(lambda *_: None)
    objc = types.SimpleNamespace(objc_msgSend=cb, sel_registerName=lambda _s: 1, objc_autoreleasePoolPush=lambda: 1, objc_autoreleasePoolPop=lambda _p: None)

    def stub(_r: Any, target: Any, sel: str | bytes, *_a: Any) -> Any:
        if sel in ("newCommandQueue", "commandBuffer"):
            seq[0] += int(sel == "newCommandQueue")
            return seq[0] if sel == "newCommandQueue" else (queues.append(target) or 500)
        return 4 if sel == "status" else (ctypes.addressof(buf) if sel == "contents" else (None if sel == "release" else 1))

    for k, v in (("_OBJC", objc), ("_DEVICE", 10), ("_COMMAND_QUEUE", 20), ("_objc_send", stub)):
        monkeypatch.setattr(me, k, v)
    monkeypatch.setattr(me, "is_metal_available", lambda: True)
    return queues


def test_use_stream_routes_metal_dispatch_to_stream_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    queues = _install_queue_spy(monkeypatch)
    stream = MetalStream(stream_id=1)
    try:
        assert stream._queue is not None and stream._queue != me._COMMAND_QUEUE
        a, w = np.ones((4, 4), dtype=np.float32), np.ones(4, dtype=np.float32)
        with use_stream(stream):
            me.metal_gemm(a, a)
            me.metal_rmsnorm(a, w)
        me.metal_gemm(a, a)
        assert queues == [stream._queue, stream._queue, me._COMMAND_QUEUE]
    finally:
        stream.release()
