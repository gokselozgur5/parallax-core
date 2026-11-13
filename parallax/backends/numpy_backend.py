from __future__ import annotations
from typing import Any
from parallax.backends.base import Backend


class NumpyBackend(Backend):
    name: str = "numpy"

    def available(self) -> bool:
        return True

    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        return op in ("gemm", "softmax", "rmsnorm")
