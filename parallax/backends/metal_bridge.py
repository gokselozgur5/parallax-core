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
