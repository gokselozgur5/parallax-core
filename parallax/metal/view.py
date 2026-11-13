from __future__ import annotations

import ctypes
from typing import Any, Optional
import numpy as np
try:
    from parallax.metal.aligned import aligned_empty, is_aligned
except ImportError:
    def is_aligned(arr: np.ndarray, page_size: Optional[int] = None) -> bool:
        ps = page_size or 4096
        return (arr.ctypes.data % ps == 0) and (arr.nbytes % ps == 0)

    def aligned_empty(shape: int | tuple[int, ...], dtype: Any = np.float32, page_size: Optional[int] = None) -> np.ndarray:
        import mmap
        ps = page_size or 4096
        dt = np.dtype(dtype)
        num_items = shape if isinstance(shape, int) else int(np.prod(shape))
        nb = max(1, num_items * dt.itemsize)
        rem = nb % ps
        alloc = nb if rem == 0 else nb + (ps - rem)
        return np.frombuffer(mmap.mmap(-1, alloc), dtype=dt)


class MetalBufferView:
    def __init__(
        self,
        array: np.ndarray,
        device_buffer: Optional[Any] = None,
        is_zero_copy: bool = False,
    ) -> None:
        self.array: np.ndarray = array
        self.device_buffer: Optional[Any] = device_buffer
        self.is_zero_copy: bool = is_zero_copy

    @property
    def ptr(self) -> int:
        return int(self.array.ctypes.data)

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes)

    def as_numpy(self) -> np.ndarray:
        return self.array

    def release(self) -> None:
        if self.device_buffer is not None:
            try:
                import parallax.metal_engine as me
                me._objc_send(None, self.device_buffer, "release")
            except Exception:
                pass
            self.device_buffer = None

    def __enter__(self) -> MetalBufferView:
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


def wrap_buffer(arr: np.ndarray, page_size: Optional[int] = None) -> MetalBufferView:
    aligned_arr = arr if is_aligned(arr, page_size) else aligned_empty(arr.shape, dtype=arr.dtype, page_size=page_size)
    if aligned_arr is not arr:
        aligned_arr[...] = arr

    dev_buf, zero_copy = None, False
    try:
        import parallax.metal_engine as me
        if me.is_metal_available():
            me._init_metal()
            if me._DEVICE is not None:
                dev_buf = me._objc_send(
                    ctypes.c_void_p,
                    me._DEVICE,
                    "newBufferWithBytesNoCopy:length:options:deallocator:",
                    ctypes.c_void_p(aligned_arr.ctypes.data),
                    int(aligned_arr.nbytes),
                    0,
                    None,
                )
                zero_copy = bool(dev_buf)
    except Exception:
        pass
    return MetalBufferView(aligned_arr, device_buffer=dev_buf, is_zero_copy=zero_copy)
