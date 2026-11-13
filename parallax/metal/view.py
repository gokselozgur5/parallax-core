from __future__ import annotations

from typing import Any, Optional
import numpy as np


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
