from typing import Any
import numpy as np

from parallax.backends import registry
from parallax.backends.base import Backend
from parallax.backends.c_backend import CBackend
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.tensor_ops import dispatched_matmul
from parallax.tensor import Tensor


class _FakeNativeBackend(Backend):
    name = "c"

    def available(self) -> bool:
        return True

    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        if op != "gemm":
            return False
        try:
            return np.dtype(dtype) == np.float32
        except Exception:
            return False

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        seen.append((a.dtype, b.dtype))
        return (a @ b).astype(np.float32)


seen: list[tuple[np.dtype, np.dtype]] = []


def test_dispatched_matmul_reaches_native_backend(monkeypatch):
    seen.clear()
    monkeypatch.setenv("PARALLAX_BACKEND", "c")
    registry.clear()
    registry.register("numpy", NumpyBackend())
    registry.register("c", _FakeNativeBackend())
    try:
        x = Tensor(np.ones((64, 128), dtype=np.float32))
        w = Tensor(np.ones((128, 64), dtype=np.float32))
        out = dispatched_matmul(x, w)
    finally:
        registry.clear()
    assert out.shape == (64, 64)
    assert seen, "PARALLAX_BACKEND=c was skipped and NumPy ran instead"


def test_dispatched_matmul_reaches_the_forced_native_backend(monkeypatch):
    backend = CBackend()
    if not backend.available():
        return
    seen_c = []
    real = CBackend.gemm
    monkeypatch.setattr(
        CBackend,
        "gemm",
        lambda s, a, b: seen_c.append(a.dtype) or real(s, a, b),
    )
    monkeypatch.setenv("PARALLAX_BACKEND", "c")
    registry.clear()
    registry.register("numpy", NumpyBackend())
    registry.register("c", backend)
    try:
        x = Tensor(np.ones((64, 128), dtype=np.float32))
        w = Tensor(np.ones((128, 64), dtype=np.float32))
        out = dispatched_matmul(x, w)
    finally:
        registry.clear()
    assert out.shape == (64, 64)
    assert seen_c, "PARALLAX_BACKEND=c was skipped and NumPy ran instead"
