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
