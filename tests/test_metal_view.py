from __future__ import annotations

import mmap
import numpy as np
from parallax.metal.view import MetalBufferView, wrap_buffer


def _alloc_aligned(size: int) -> np.ndarray:
    mm = mmap.mmap(-1, max(16384, size * 4))
    return np.frombuffer(mm, dtype=np.float32)


def test_metal_buffer_view_properties() -> None:
    arr = _alloc_aligned(256)
    view = MetalBufferView(arr)
    assert view.ptr == int(arr.ctypes.data)
    assert view.nbytes == int(arr.nbytes)
    assert np.shares_memory(view.as_numpy(), arr)


def test_metal_buffer_view_context_manager() -> None:
    arr = _alloc_aligned(128)
    with wrap_buffer(arr) as view:
        assert view.ptr == int(arr.ctypes.data)
        assert np.all(view.as_numpy() == 0)
    assert view.device_buffer is None


def test_wrap_buffer_mutation_reflected() -> None:
    arr = _alloc_aligned(64)
    view = wrap_buffer(arr)
    view.as_numpy()[0] = 999.0
    assert arr[0] == np.float32(999.0)
    view.release()


def test_wrap_buffer_unaligned_copy() -> None:
    unaligned = np.ones((5,), dtype=np.float32)
    view = wrap_buffer(unaligned)
    assert np.all(view.as_numpy()[:5] == 1.0)
    view.release()
