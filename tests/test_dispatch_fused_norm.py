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


