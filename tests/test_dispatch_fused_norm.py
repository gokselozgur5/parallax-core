"""Tests for Fused RMSNorm dispatch with epsilon stabilization."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor

from python.parallax.dispatch.fused_norm import (
    FusedRMSNorm,
    fused_rmsnorm,
    stabilize_epsilon,
)



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


def test_fused_rmsnorm_module_init_and_params():
    norm = FusedRMSNorm(dim=16, eps=1e-5)
    assert norm.dim == 16
    assert norm.eps == 1e-5
    assert norm.weight.shape == (16,)
    np.testing.assert_allclose(norm.weight.data, 1.0)


def test_fused_rmsnorm_module_forward():
    norm = FusedRMSNorm(dim=4, eps=1e-5)
    x = np.ones((2, 4), dtype=np.float32)
    out = norm(x)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out.data, np.full((2, 4), 1.0, dtype=np.float32), rtol=1e-4)


def test_fused_rmsnorm_repr_and_from_norm():
    norm = FusedRMSNorm(dim=32, eps=1e-6)
    assert "dim=32" in repr(norm)
    assert "eps=1e-06" in repr(norm)
    cloned = FusedRMSNorm.from_norm(norm)
    assert cloned.dim == 32
    assert cloned.eps == 1e-6


def test_fused_rmsnorm_autograd_backward():
    norm = FusedRMSNorm(dim=4, eps=1e-5)
    x = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), requires_grad=True)
    res = Tensor(np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32), requires_grad=True)
    out = norm(x, residual=res)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert res.grad is not None
    assert norm.weight.grad is not None
    np.testing.assert_allclose(x.grad, res.grad)


def test_fused_norm_exports():
    import python.parallax.dispatch.fused_norm as fn
    assert hasattr(fn, "FusedRMSNorm")
    assert hasattr(fn, "fused_rmsnorm")
    assert hasattr(fn, "stabilize_epsilon")


def test_fused_norm_package_exports():
    import parallax.dispatch as pd
    assert hasattr(pd, "FusedRMSNorm")
    assert hasattr(pd, "fused_rmsnorm")
    assert hasattr(pd, "stabilize_epsilon")













