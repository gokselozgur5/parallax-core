from __future__ import annotations

from typing import Any
import numpy as np

from parallax.backends.base import Backend
from parallax.rust_engine import is_rust_engine_available


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
            return np.dtype(dtype) == np.float32
        except Exception:
            return False
