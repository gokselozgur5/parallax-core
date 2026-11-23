"""Regression tests for epsilon stabilization in FusedLayerNorm backward."""

import numpy as np
import pytest
from parallax.dispatch import DispatchedFusedLayerNorm, FusedLayerNorm
from parallax.tensor import Tensor


def test_fused_layernorm_zero_eps_constant_row_finite_grad() -> None:
    norm = FusedLayerNorm(8, eps=0.0)
    x = Tensor(np.ones((1, 8), dtype=np.float32), requires_grad=True)
    dout = np.arange(8, dtype=np.float32).reshape(1, 8)
    out = norm(x)
    out.backward(dout)
    assert x.grad is not None and norm.weight is not None and norm.weight.grad is not None
    assert np.all(np.isfinite(x.grad))
    np.testing.assert_allclose(x.grad, 1e6 * (dout - dout.mean()), rtol=1e-5)
    np.testing.assert_allclose(norm.weight.grad, np.zeros(8, dtype=np.float32))


def test_fused_layernorm_tiny_eps_backward_matches_stabilized_eps() -> None:
    norm = FusedLayerNorm(4, eps=0.0, elementwise_affine=False)
    x_np = np.array([[0.0, 1e-8, -1e-8, 0.0]], dtype=np.float64)
    dout = np.array([[1.0, -2.0, 0.5, 0.5]], dtype=np.float64)
    x = Tensor(x_np, requires_grad=True)
    norm(x).backward(dout)
    inv_std = 1.0 / np.sqrt(np.var(x_np, axis=-1, keepdims=True) + 1e-12)
    x_hat = (x_np - np.mean(x_np, axis=-1, keepdims=True)) * inv_std
    expected = inv_std * (dout - dout.mean(axis=-1, keepdims=True) - x_hat * np.mean(dout * x_hat))
    np.testing.assert_allclose(x.grad, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("cls", [FusedLayerNorm, DispatchedFusedLayerNorm])
def test_tuple_normalized_shape_forward_and_backward(cls: type) -> None:
    x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4) ** 1.5
    mean = np.mean(x_np, axis=(-2, -1), keepdims=True)
    var = np.var(x_np, axis=(-2, -1), keepdims=True)
    expected = (x_np - mean) / np.sqrt(var + 1e-5)

    layer = cls((3, 4), eps=1e-5)
    x = Tensor(x_np.copy(), requires_grad=True)
    out = layer(x)
    np.testing.assert_allclose(out.data, expected, rtol=1e-6, atol=1e-6)
    out.sum().backward()
    assert layer.weight is not None and layer.weight.grad is not None
    assert layer.bias is not None and layer.bias.grad is not None
    assert layer.weight.grad.shape == (3, 4)
    np.testing.assert_allclose(layer.bias.grad, np.full((3, 4), 2.0))


@pytest.mark.parametrize("cls", [FusedLayerNorm, DispatchedFusedLayerNorm])
def test_tuple_normalized_shape_no_affine_gradcheck_and_mismatch(cls: type) -> None:
    rng = np.random.default_rng(7)
    x_np = rng.normal(size=(2, 3, 4))
    dout = rng.normal(size=(2, 3, 4))
    layer = cls((3, 4), eps=1e-5, elementwise_affine=False)
    x = Tensor(x_np.copy(), requires_grad=True)
    out = layer(x)
    out.backward(dout)

    def ref_loss(arr: np.ndarray) -> float:
        m = np.mean(arr, axis=(-2, -1), keepdims=True)
        v = np.var(arr, axis=(-2, -1), keepdims=True)
        return float(np.sum(((arr - m) / np.sqrt(v + 1e-5)) * dout))

    fd = np.zeros_like(x_np)
    for idx in np.ndindex(x_np.shape):
        pos, neg = x_np.copy(), x_np.copy()
        pos[idx] += 1e-6
        neg[idx] -= 1e-6
        fd[idx] = (ref_loss(pos) - ref_loss(neg)) / 2e-6
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, fd, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        layer(Tensor(np.zeros((2, 3, 5))))

