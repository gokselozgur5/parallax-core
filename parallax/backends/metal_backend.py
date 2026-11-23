from __future__ import annotations

import sys
from typing import Any
import numpy as np
from parallax.backends.base import Backend
import parallax.metal_engine as _metal_engine


def is_metal_available() -> bool:
    if sys.platform != "darwin":
        return False
    return bool(_metal_engine.is_metal_available())


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
        if op not in ("gemm", "rmsnorm"):
            return False
        try:
            if np.dtype(dtype) != np.float32:
                return False
        except Exception:
            return False
        if op == "gemm":
            return len(shape) == 2
        if op == "rmsnorm":
            return len(shape) >= 1
        return True

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.available():
            raise RuntimeError("Metal backend is not available")
        return _metal_engine.metal_gemm(a, b)

    def rmsnorm(
        self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        if not self.available():
            raise RuntimeError("Metal backend is not available")
        w = np.squeeze(weight) if weight.ndim > 1 else weight
        return _metal_engine.metal_rmsnorm(x, w, eps=eps)
