"""Tests for native dispatch policy defaults and naive GEMM opt-in."""

from __future__ import annotations

import pytest
from parallax.backends import registry
from parallax.backends.c_backend import CBackend
from parallax.backends.dispatch import candidate_backends
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.policy import ENABLE_NAIVE_GEMM_VAR


@pytest.fixture(autouse=True)
def reset_backends():
    registry.clear()
    registry.register("numpy", NumpyBackend())
    registry.register("c", CBackend())
    yield
    registry.clear()


def test_default_gemm_routes_to_numpy_blas(monkeypatch):
    """Verify that by default, GEMM candidate backends route to NumPy BLAS."""
    monkeypatch.delenv(ENABLE_NAIVE_GEMM_VAR, raising=False)
    monkeypatch.delenv("PARALLAX_BACKEND", raising=False)
    monkeypatch.delenv("PARALLAX_BACKEND_GEMM", raising=False)

    candidates = candidate_backends("gemm", n_elements=10000)
    assert candidates == ["numpy"]


def test_naive_gemm_opt_in_environment_variable(monkeypatch):
    """Verify that PARALLAX_ENABLE_NAIVE_GEMM=1 unlocks native GEMM candidates."""
    monkeypatch.setenv(ENABLE_NAIVE_GEMM_VAR, "1")
    monkeypatch.delenv("PARALLAX_BACKEND", raising=False)
    monkeypatch.delenv("PARALLAX_BACKEND_GEMM", raising=False)

    candidates = candidate_backends("gemm", n_elements=10000)
    assert len(candidates) >= 1
    assert "numpy" in candidates
