from __future__ import annotations

import numpy as np
from parallax.backends.base import check_backend_contract
from parallax.backends.c_backend import CBackend
from parallax.c_engine import is_c_engine_available


def test_c_backend_contract() -> None:
    backend = CBackend()
    check_backend_contract(backend)
    assert backend.name == "c"


def test_c_backend_availability() -> None:
    backend = CBackend()
    assert backend.available() == is_c_engine_available()


def test_c_backend_supports() -> None:
    backend = CBackend()
    if not backend.available():
        assert not backend.supports("gemm", np.float32, (2, 2))
        return
    assert backend.supports("gemm", np.float32, (2, 2))
    assert backend.supports("softmax", "float32", (2, 2))
    assert backend.supports("rmsnorm", np.float32, (2, 2))
    assert not backend.supports("gemm", np.float64, (2, 2))
    assert not backend.supports("unknown_op", np.float32, (2, 2))
