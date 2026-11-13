from __future__ import annotations

from typing import Any
import numpy as np
from parallax.backends.base import Backend
from parallax.c_engine import (
    c_gemm,
    c_rmsnorm,
    c_softmax,
    is_c_engine_available,
)


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

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return c_gemm(a, b)

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        return c_softmax(x, axis=axis)

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        return c_rmsnorm(x, weight, eps=eps)
