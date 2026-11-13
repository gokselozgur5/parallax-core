from __future__ import annotations

from typing import Any
import numpy as np
from parallax.backends.base import Backend
from parallax.c_engine import is_c_engine_available


class CBackend(Backend):
    name: str = "c"

    def available(self) -> bool:
        return is_c_engine_available()

    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        if not self.available():
            return False
        if op not in ("gemm", "softmax", "rmsnorm"):
            return False
        try:
            return np.dtype(dtype) == np.float32
        except Exception:
            return False
