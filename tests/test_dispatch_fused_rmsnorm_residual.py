"""Tests for fused RMSNorm forward and backward residual kernels."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from python.parallax.dispatch.fused_rmsnorm_residual import (
    FusedRMSNormResidual,
    dispatched_fused_rmsnorm_residual,
    fused_rmsnorm_residual_backward,
    fused_rmsnorm_residual_backward_reference,
    fused_rmsnorm_residual_forward,
    fused_rmsnorm_residual_forward_reference,
    stabilize_epsilon,
)


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-5.0) == 1e-12


def test_fused_rmsnorm_residual_forward_shapes():
    np.random.seed(42)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    res = np.random.randn(2, 4, 8).astype(np.float32)
    w = np.ones(8, dtype=np.float32)
    normed, fused = fused_rmsnorm_residual_forward_reference(x, res, w, eps=1e-5)
    assert normed.shape == x.shape
    assert fused.shape == res.shape
    np.testing.assert_allclose(fused, x + res)
    assert not np.isnan(normed).any()
    assert not np.isnan(fused).any()
