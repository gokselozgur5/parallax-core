from __future__ import annotations

import sys
import numpy as np
import pytest
from parallax.backends.base import check_backend_contract
from parallax.backends.metal_backend import MetalBackend
from parallax.metal_engine import is_metal_available


def test_metal_backend_contract() -> None:
    backend = MetalBackend()
    check_backend_contract(backend)
    assert backend.name == "metal"


def test_metal_backend_availability() -> None:
    backend = MetalBackend()
    if sys.platform != "darwin":
        assert not backend.available()
    else:
        assert backend.available() == is_metal_available()


def test_metal_backend_supports() -> None:
    backend = MetalBackend()
    if not backend.available():
        assert not backend.supports("gemm", np.float32, (2, 2))
        return
    assert backend.supports("gemm", np.float32, (2, 2))
    assert backend.supports("rmsnorm", "float32", (2, 2))
    assert not backend.supports("softmax", np.float32, (2, 2))
    assert not backend.supports("gemm", np.float64, (2, 2))
    assert not backend.supports("unknown_op", np.float32, (2, 2))


def test_metal_backend_unavailable_off_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    backend = MetalBackend()
    assert not backend.available()
    assert not backend.supports("gemm", np.float32, (2, 2))


def test_metal_backend_parity_with_numpy() -> None:
    backend = MetalBackend()
    if not backend.available():
        pytest.skip("Metal backend is not available on this platform")
    from parallax.backends.numpy_backend import NumpyBackend

    np_backend = NumpyBackend()
    a = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(8, 6).astype(np.float32)
    assert np.allclose(backend.gemm(a, b), np_backend.gemm(a, b), atol=1e-4)

    x = np.random.randn(4, 16).astype(np.float32)
    w = np.ones((16,), dtype=np.float32)
    assert np.allclose(
        backend.rmsnorm(x, w), np_backend.rmsnorm(x, w), atol=1e-4
    )
