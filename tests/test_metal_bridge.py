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


def test_zero_copy_bridge_and_view_cache_share_memory(monkeypatch) -> None:
    import ctypes
    import gc
    import parallax.metal_engine as me
    from parallax.backends.metal_pool import ZeroCopyViewCache

    live_bridge, live_cache = MetalTensorBridge(), ZeroCopyViewCache()
    if live_bridge.available():
        live_arr = np.arange(8, dtype=np.float32)
        try:
            assert live_bridge.is_shared(live_arr, live_cache.get_or_create(live_arr))
            live_arr[:] = 99.0
            seen = live_bridge.from_metal_buffer(live_cache.get_or_create(live_arr), live_arr.shape)
            np.testing.assert_array_equal(seen, live_arr)
        finally:
            live_cache.clear()

    released: list[object] = []

    def make_buf(ptr_arg, n: int, copy: bool):
        p = ptr_arg.value if isinstance(ptr_arg, ctypes.c_void_p) else int(ptr_arg)
        raw = (ctypes.c_char * n).from_buffer_copy((ctypes.c_char * n).from_address(p)) if copy else None
        return type("Buf", (), {"_raw": raw, "contents": ctypes.addressof(raw) if raw else p, "length": n})()

    def sim_objc(restype, target, sel, *args):
        if sel == "newBufferWithBytes:length:options:":
            return make_buf(args[0], int(args[1]), copy=True)
        if sel == "newBufferWithBytesNoCopy:length:options:deallocator:":
            return make_buf(args[0], int(args[1]), copy=False)
        if sel == "release":
            released.append(target)
        return getattr(target, sel, None) if sel in ("length", "contents") else None

    monkeypatch.setattr(me, "_DEVICE", 1)
    monkeypatch.setattr(me, "_objc_send", sim_objc)
    bridge, cache = MetalTensorBridge(), ZeroCopyViewCache()
    monkeypatch.setattr(bridge, "available", lambda: True)
    monkeypatch.setattr(cache, "available", lambda: True)
    backing = bytearray(32)
    arr = np.ndarray((8,), dtype=np.float32, buffer=backing)
    arr[:] = np.arange(8, dtype=np.float32)
    b_bridge, shape, dt = bridge.to_metal_buffer(arr)
    assert bridge.is_shared(arr, b_bridge)
    b1 = cache.get_or_create(arr)
    arr[:] = 99.0
    np.testing.assert_array_equal(bridge.from_metal_buffer(b_bridge, shape, dt), arr)
    np.testing.assert_array_equal(bridge.from_metal_buffer(cache.get_or_create(arr), shape, dt), arr)
    bridge.release_buffer(b_bridge)
    del arr
    gc.collect()
    assert cache.count() == 0 and b1 in released
    arr2 = np.ndarray((8,), dtype=np.float32, buffer=backing)
    arr2[:] = 7.0
    assert cache.get_or_create(arr2) is not b1
    np.testing.assert_array_equal(bridge.from_metal_buffer(cache.get_or_create(arr2), shape, dt), arr2)

