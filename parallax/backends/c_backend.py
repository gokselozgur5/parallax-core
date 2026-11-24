from __future__ import annotations

from typing import Any
import numpy as np
from parallax.backends.base import Backend
from parallax.c_engine import (
    _find_or_build_library,
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
            dt = np.dtype(dtype)
        except Exception:
            return False
        if op == "rmsnorm" and dt == np.float64:
            lib = _find_or_build_library()
            return bool(lib is not None and hasattr(lib, "c_rmsnorm_f64") and len(shape) >= 1)
        if dt != np.float32:
            return False
        if op == "gemm":
            return len(shape) == 2
        if op in ("softmax", "rmsnorm"):
            return len(shape) >= 1
        return True

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return c_gemm(a, b)

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        return c_softmax(x, axis=axis)

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        w = np.squeeze(weight) if weight.ndim > 1 else weight
        return c_rmsnorm(x, w, eps=eps)


CEngineBackend = CBackend
