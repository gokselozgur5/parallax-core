from __future__ import annotations

import sys
from typing import Any
import numpy as np
from parallax.backends.base import Backend


def is_metal_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        from parallax.metal_engine import is_metal_available as _avail

        return bool(_avail())
    except Exception:
        return False


def probe_metal() -> dict[str, Any]:
    avail = is_metal_available()
    return {
        "available": avail,
        "platform": sys.platform,
        "device_name": "Apple Metal GPU" if avail else None,
    }


class MetalBackend(Backend):
    name: str = "metal"

    def available(self) -> bool:
        return is_metal_available()

    def probe(self) -> dict[str, Any]:
        return probe_metal()

    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        if not self.available():
            return False
        if op not in ("gemm", "rmsnorm", "softmax"):
            return False
        try:
            return np.dtype(dtype) == np.float32
        except Exception:
            return False

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.available():
            raise RuntimeError("Metal backend is not available")
        from parallax.metal_engine import metal_gemm

        return metal_gemm(a, b)

    def rmsnorm(
        self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        if not self.available():
            raise RuntimeError("Metal backend is not available")
        from parallax.metal_engine import metal_rmsnorm

        return metal_rmsnorm(x, weight, eps=eps)

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        if not self.available():
            raise RuntimeError("Metal backend is not available")
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
