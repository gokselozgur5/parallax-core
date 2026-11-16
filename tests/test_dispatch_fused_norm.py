"""Tests for Fused RMSNorm dispatch with epsilon stabilization."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

import parallax

REPO_ROOT = Path(__file__).resolve().parent.parent
_pkg = str(REPO_ROOT / "python" / "parallax")
if _pkg not in parallax.__path__:
    parallax.__path__.append(_pkg)

from parallax.dispatch.fused_norm import fused_rmsnorm, stabilize_epsilon


def test_stabilize_epsilon_positive_value():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(1e-6) == 1e-6


def test_stabilize_epsilon_clamping():
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-1.0) == 1e-12


def test_fused_rmsnorm_forward_numpy_parity():
    np.random.seed(42)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    w = np.ones(8, dtype=np.float32)
    out = fused_rmsnorm(x, w, eps=1e-5)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    expected = (x / rms) * w
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_fused_rmsnorm_with_residual_addition():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    res = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float32)
    w = np.ones(2, dtype=np.float32)
    normed, fused = fused_rmsnorm(x, w, residual=res, return_residual=True)
    np.testing.assert_allclose(fused, x + res)
    expected = fused_rmsnorm(x + res, w)
    np.testing.assert_allclose(normed, expected)


def test_fused_rmsnorm_dispatches_to_backend():
    x = np.ones((2, 4), dtype=np.float32)
    w = np.full(4, 2.0, dtype=np.float32)
    out = fused_rmsnorm(x, w, eps=1e-5)
    np.testing.assert_allclose(out, np.full((2, 4), 2.0, dtype=np.float32), rtol=1e-4)


def test_fused_rmsnorm_zero_inputs_stability():
    x = np.zeros((2, 4), dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    out = fused_rmsnorm(x, w, eps=1e-5)
    np.testing.assert_allclose(out, 0.0)
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))






