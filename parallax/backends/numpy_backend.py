from __future__ import annotations
from typing import Any
import numpy as np
from parallax.backends.base import Backend


class NumpyBackend(Backend):
    name: str = "numpy"

    def available(self) -> bool:
        return True

    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        return op in ("gemm", "softmax", "rmsnorm")

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        variance = np.mean(x * x, axis=-1, keepdims=True)
        return (x / np.sqrt(variance + eps)) * weight
