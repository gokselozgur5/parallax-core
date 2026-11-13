from __future__ import annotations

import sys
from typing import Any
import numpy as np
from parallax.backends.base import Backend
from parallax.metal_engine import is_metal_available, metal_gemm, metal_rmsnorm


class MetalBackend(Backend):
    name: str = "metal"

    def available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return is_metal_available()

    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        if not self.available():
            return False
        if op not in ("gemm", "rmsnorm"):
            return False
        try:
            return np.dtype(dtype) == np.float32
        except Exception:
            return False

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return metal_gemm(a, b)

    def rmsnorm(
        self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        return metal_rmsnorm(x, weight, eps=eps)
