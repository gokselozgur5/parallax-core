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
