from __future__ import annotations

import numpy as np
import pytest
from parallax.backends.base import check_backend_contract
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.rust_backend import RustBackend
from parallax.rust_engine import is_rust_engine_available


def test_rust_backend_contract() -> None:
    backend = RustBackend()
    check_backend_contract(backend)
    assert backend.name == "rust"


def test_rust_backend_availability() -> None:
    backend = RustBackend()
    assert backend.available() == is_rust_engine_available()


def test_rust_backend_supports() -> None:
    backend = RustBackend()
    if not backend.available():
        assert not backend.supports("gemm", np.float32, (2, 2))
        return
    assert backend.supports("gemm", np.float32, (2, 2))
    assert backend.supports("softmax", "float32", (2, 2))
    assert not backend.supports("rmsnorm", np.float32, (2, 2))
    assert not backend.supports("gemm", np.float64, (2, 2))
    assert not backend.supports("unknown_op", np.float32, (2, 2))


def test_rust_backend_parity_with_numpy() -> None:
    backend = RustBackend()
    if not backend.available():
        pytest.skip("Rust engine library not available")
    np_backend = NumpyBackend()
    a = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(8, 6).astype(np.float32)
    assert np.allclose(backend.gemm(a, b), np_backend.gemm(a, b), atol=1e-4)

    x = np.random.randn(4, 16).astype(np.float32)
    assert np.allclose(backend.softmax(x), np_backend.softmax(x), atol=1e-4)
