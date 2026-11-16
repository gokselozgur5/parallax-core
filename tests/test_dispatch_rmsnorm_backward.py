"""Tests for RMSNorm backward autograd functional kernel."""

from __future__ import annotations

import numpy as np
import pytest

from parallax.tensor import Tensor
from python.parallax.dispatch.rmsnorm_backward import (
    rmsnorm_backward_reference,
    stabilize_epsilon,
)


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-5.0) == 1e-12


def test_rmsnorm_backward_shapes_and_values():
    np.random.seed(42)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    w = np.ones(8, dtype=np.float32)
    dout = np.ones_like(x)
    dx, dw = rmsnorm_backward_reference(dout, x, w, eps=1e-5)
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert not np.isnan(dx).any()
    assert not np.isnan(dw).any()
