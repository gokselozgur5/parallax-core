"""Metal unified memory zero-copy tensor bridge."""

from __future__ import annotations

import sys
from typing import Any, Optional
import numpy as np

from parallax.metal_engine import is_metal_available


class MetalTensorBridge:
    """Bridge between Python/NumPy/Tensor data and Metal shared buffers."""

    def available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return is_metal_available()

    def unwrap_array(self, obj: Any) -> np.ndarray:
        data = getattr(obj, "data", obj)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)
        return np.ascontiguousarray(data)

    def to_metal_buffer(
        self, obj: Any
    ) -> tuple[Optional[Any], tuple[int, ...], np.dtype]:
        arr = self.unwrap_array(obj)
        if not self.available() or arr.size == 0:
            return None, arr.shape, arr.dtype
        import parallax.metal_engine as me

        if me._DEVICE is None or me._objc_send is None:
            return None, arr.shape, arr.dtype
        buf = me._objc_send(
            me.ctypes.c_void_p,
            me._DEVICE,
            "newBufferWithBytes:length:options:",
            arr.ctypes.data_as(me.ctypes.c_void_p),
            arr.nbytes,
            0,
        )
        return buf, arr.shape, arr.dtype

    def release_buffer(self, buffer: Any) -> None:
        import parallax.metal_engine as me

        try:
            if me._objc_send is not None and buffer:
                me._objc_send(None, buffer, "release")
        except Exception:
            pass

    def from_metal_buffer(
        self,
        buffer: Any,
        shape: tuple[int, ...],
        dtype: Any = np.float32,
    ) -> Optional[np.ndarray]:
        if not self.available() or buffer is None:
            return None
        import ctypes
        import parallax.metal_engine as me

        ptr = me._objc_send(ctypes.c_void_p, buffer, "contents")
        if not ptr:
            return None
        c_type = ctypes.POINTER(ctypes.c_float)
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, c_type), shape=shape)
        return arr.copy()

    def is_shared(self, array: np.ndarray, buffer: Any) -> bool:
        if not self.available() or buffer is None or array is None:
            return False
        import ctypes
        import parallax.metal_engine as me

        ptr = me._objc_send(ctypes.c_void_p, buffer, "contents")
        return int(array.ctypes.data) == int(ptr or 0)
