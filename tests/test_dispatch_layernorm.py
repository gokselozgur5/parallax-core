"""Tests for FusedLayerNorm layer and dispatch layernorm routines."""

import numpy as np
from python.parallax.dispatch.layernorm import FusedLayerNorm, stabilize_epsilon


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-5.0) == 1e-12


def test_fused_layernorm_initialization():
    norm = FusedLayerNorm(normalized_shape=64, eps=1e-5, elementwise_affine=True, bias=True)
    assert norm.normalized_shape == (64,)
    assert norm.eps == 1e-5
    assert norm.elementwise_affine is True
    assert norm.weight is not None
    assert norm.weight.shape == (64,)
    assert norm.bias is not None
    assert norm.bias.shape == (64,)


def test_fused_layernorm_repr():
    norm = FusedLayerNorm(normalized_shape=(16, 32), eps=1e-6, elementwise_affine=False)
    assert norm.weight is None
    assert norm.bias is None
    expected_extra = "normalized_shape=(16, 32), eps=1e-06, elementwise_affine=False"
    assert norm.extra_repr() == expected_extra
    assert repr(norm) == f"FusedLayerNorm({expected_extra})"


def test_fused_layernorm_package_exports():
    from parallax.dispatch.layernorm import FusedLayerNorm as LN1
    from python.parallax.dispatch import FusedLayerNorm as LN2

    assert LN1 is FusedLayerNorm
    assert LN2 is FusedLayerNorm


def test_fused_layernorm_forward_parity():
    from parallax.nn import LayerNorm
    from parallax.tensor import Tensor

    dim = 16
    ref_norm = LayerNorm(dim=dim, eps=1e-5)
    fused = FusedLayerNorm(normalized_shape=dim, eps=1e-5)
    fused.weight.data = ref_norm.weight.data.copy()
    fused.bias.data = ref_norm.bias.data.copy()

    x_np = np.random.randn(2, 4, dim).astype(np.float32)
    x1, x2 = Tensor(x_np.copy()), Tensor(x_np.copy())
    out1 = ref_norm(x1)
    out2 = fused(x2)
    assert np.allclose(out1.data, out2.data, atol=1e-5)


def test_fused_layernorm_residual_addition():
    from parallax.tensor import Tensor
    from python.parallax.dispatch.layernorm import fused_layernorm

    dim = 8
    x = np.random.randn(2, dim).astype(np.float32)
    res = np.random.randn(2, dim).astype(np.float32)
    w = np.ones(dim, dtype=np.float32)
    b = np.zeros(dim, dtype=np.float32)

    out, residual_out = fused_layernorm(x, w, b, residual=res, return_residual=True)
    assert np.allclose(residual_out, x + res)
    assert out.shape == x.shape

    norm = FusedLayerNorm(dim)
    t_out = norm(Tensor(x), residual=Tensor(res))
    assert t_out.shape == (2, dim)


def test_fused_layernorm_backward_gradients():
    from parallax.tensor import Tensor

    dim = 8
    rng = np.random.default_rng(42)
    norm = FusedLayerNorm(dim)
    norm.weight.data = rng.normal(size=dim).astype(np.float32)
    norm.bias.data = rng.normal(size=dim).astype(np.float32)
    x_np, res_np, c_np = rng.normal(size=(3, 2, 4, dim)).astype(np.float32)
    x = Tensor(x_np.copy(), requires_grad=True)
    res = Tensor(res_np.copy(), requires_grad=True)

    out = norm(x, residual=res)
    loss = (out * Tensor(c_np)).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 4, dim)
    assert res.grad is not None
    assert res.grad.shape == (2, 4, dim)
    assert np.allclose(x.grad, res.grad)
    assert norm.weight.grad is not None
    assert norm.weight.grad.shape == (dim,)
    assert norm.bias.grad is not None
    assert norm.bias.grad.shape == (dim,)

    p64 = [a.astype(np.float64) for a in (x_np, res_np, norm.weight.data, norm.bias.data)]
    c64 = c_np.astype(np.float64)

    def ref_loss() -> float:
        z = p64[0] + p64[1]
        normed = (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + norm.eps)
        return float(np.sum((normed * p64[2] + p64[3]) * c64))

    for grad, arr in zip((x.grad, res.grad, norm.weight.grad, norm.bias.grad), p64):
        fd = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + 1e-5
            pos = ref_loss()
            arr[idx] = orig - 1e-5
            fd[idx] = (pos - ref_loss()) / 2e-5
            arr[idx] = orig
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-4)


def test_from_layernorm_conversion():
    from parallax.nn import LayerNorm

    orig = LayerNorm(dim=12, eps=1e-4)
    orig.weight.data = np.random.randn(12).astype(np.float32)
    orig.bias.data = np.random.randn(12).astype(np.float32)

    converted = FusedLayerNorm.from_layernorm(orig)
    assert converted.normalized_shape == (12,)
    assert converted.eps == 1e-4
    assert np.allclose(converted.weight.data, orig.weight.data)
    assert np.allclose(converted.bias.data, orig.bias.data)
