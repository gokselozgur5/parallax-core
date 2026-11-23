from __future__ import annotations

from typing import Any
import numpy as np

from parallax.backends.base import Backend
from parallax.rust_engine import (
    is_rust_engine_available,
    rust_gemm,
    rust_softmax,
)


class RustBackend(Backend):
    name: str = "rust"

    def available(self) -> bool:
        return is_rust_engine_available()

    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        if not self.available():
            return False
        if op not in ("gemm", "softmax"):
            return False
        try:
            if np.dtype(dtype) != np.float32:
                return False
        except Exception:
            return False
        if op == "gemm":
            return len(shape) == 2
        if op == "softmax":
            return len(shape) >= 1
        return True

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return rust_gemm(a, b)

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        return rust_softmax(x, axis=axis)


RustEngineBackend = RustBackend
