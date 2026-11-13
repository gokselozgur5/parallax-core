from __future__ import annotations

import numpy as np
from parallax.backends.base import check_backend_contract
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
