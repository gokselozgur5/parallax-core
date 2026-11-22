from parallax.backends.metal_bridge import MetalTensorBridge
from parallax.tensor import Tensor
import numpy as np


class FakeBuffer:
    def __init__(self, data: bytes) -> None:
        import ctypes
        self._raw = (ctypes.c_char * len(data)).from_buffer_copy(data)
        self.contents = ctypes.cast(self._raw, ctypes.c_void_p).value
        self.length = len(data)


def test_from_metal_buffer_dtype(monkeypatch) -> None:
    bridge = MetalTensorBridge()
    monkeypatch.setattr(bridge, "available", lambda: True)
    t = Tensor([1.0, 2.0])  # float64
    fake = FakeBuffer(t.data.tobytes())

    import parallax.metal_engine as me

    def fake_objc(restype, target, sel, *args):
        if sel == "length":
            return getattr(target, "length", 0)
        if sel == "contents":
            return getattr(target, "contents", None)
        return None

    monkeypatch.setattr(me, "_objc_send", fake_objc)

    rec = bridge.from_metal_buffer(fake, t.shape, t.data.dtype)
    assert rec is not None
    assert np.allclose(t.data, rec)
