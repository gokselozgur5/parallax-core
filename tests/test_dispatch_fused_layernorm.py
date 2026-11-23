"""Tests for Dispatched LayerNorm with RMSNorm weight scaling and bias fusion."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from parallax.dispatch.fused_layernorm import (
    DispatchedFusedLayerNorm,
    FusedLayerNormRMS,
    dispatched_fused_layernorm,
    fused_layernorm_backward,
    fused_layernorm_forward,
    rms_weight_scale,
    rms_weight_scale_backward,
    stabilize_epsilon,
)


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-2.0) == 1e-12


def test_rms_weight_scale_and_backward():
    np.random.seed(42)
    w = np.random.randn(8).astype(np.float64)
    dout = np.random.randn(8).astype(np.float64)
    scaled = rms_weight_scale(w)
    expected_scale = w / np.sqrt(np.mean(w ** 2) + 1e-6)
    np.testing.assert_allclose(scaled, expected_scale, rtol=1e-5)
    ana_dw = rms_weight_scale_backward(dout, w)
    h = 1e-6
    num_dw = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy(); w_p[i] += h
        w_m = w.copy(); w_m[i] -= h
        num_dw[i] = np.sum((rms_weight_scale(w_p) - rms_weight_scale(w_m)) / (2 * h) * dout)
    np.testing.assert_allclose(ana_dw, num_dw, rtol=1e-5, atol=1e-6)


def test_fused_layernorm_forward_shapes_and_parity():
    np.random.seed(99)
    x = np.random.randn(2, 4, 16).astype(np.float32)
    w = np.ones(16, dtype=np.float32)
    b = np.zeros(16, dtype=np.float32)
    out = fused_layernorm_forward(x, w, b, eps=1e-5)
    assert out.shape == (2, 4, 16)
    mean = np.mean(out, axis=-1)
    std = np.std(out, axis=-1)
    np.testing.assert_allclose(mean, np.zeros((2, 4)), atol=1e-5)
    np.testing.assert_allclose(std, np.ones((2, 4)), atol=1e-3)


def test_fused_layernorm_residual_and_bias():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    res = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    b = np.array([0.1, -0.1], dtype=np.float32)
    out, out_res = fused_layernorm_forward(x, bias=b, residual=res, return_residual=True)
    np.testing.assert_allclose(out_res, x + res)
    fused = x + res
    mean = np.mean(fused, axis=-1, keepdims=True)
    var = np.var(fused, axis=-1, keepdims=True)
    expected = (fused - mean) / np.sqrt(var + 1e-5) + b
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_fused_layernorm_rms_weight_scale():
    x = np.array([[2.0, 4.0], [1.0, 3.0]], dtype=np.float32)
    w = np.array([2.0, 2.0], dtype=np.float32)
    out_std = fused_layernorm_forward(x, weight=w, rms_scale=False)
    out_rms = fused_layernorm_forward(x, weight=w, rms_scale=True)
    expected_w = w / np.sqrt(np.mean(w ** 2) + 1e-5)
    np.testing.assert_allclose(expected_w, np.array([1.0, 1.0], dtype=np.float32), rtol=1e-3)
    np.testing.assert_allclose(out_rms * 2.0, out_std, rtol=1e-3)


def test_fused_layernorm_backward_analytical_dx():
    np.random.seed(123)
    x = np.random.randn(2, 3, 4).astype(np.float64)
    res = np.random.randn(2, 3, 4).astype(np.float64)
    w = np.random.randn(4).astype(np.float64)
    b = np.random.randn(4).astype(np.float64)
    dout = np.random.randn(2, 3, 4).astype(np.float64)
    for rms in (False, True):
        dx_ana, _, _, dres_ana = fused_layernorm_backward(dout, x, w, b, res, eps=1e-5, rms_scale=rms)
        h = 1e-6
        num_dx = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x_p = x.copy(); x_p[i, j, k] += h
                    x_m = x.copy(); x_m[i, j, k] -= h
                    f_p = fused_layernorm_forward(x_p, w, b, res, eps=1e-5, rms_scale=rms)
                    f_m = fused_layernorm_forward(x_m, w, b, res, eps=1e-5, rms_scale=rms)
                    num_dx[i, j, k] = np.sum((f_p - f_m) / (2 * h) * dout)
        np.testing.assert_allclose(dx_ana, num_dx, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dres_ana, num_dx, rtol=1e-5, atol=1e-6)


def test_fused_layernorm_backward_analytical_params():
    np.random.seed(456)
    x = np.random.randn(2, 4).astype(np.float64)
    w = np.random.randn(4).astype(np.float64)
    b = np.random.randn(4).astype(np.float64)
    dout = np.random.randn(2, 4).astype(np.float64)
    for rms in (False, True):
        _, dw_ana, db_ana, _ = fused_layernorm_backward(dout, x, w, b, eps=1e-5, rms_scale=rms)
        h = 1e-6
        num_dw = np.zeros_like(w)
        for i in range(len(w)):
            w_p = w.copy(); w_p[i] += h
            w_m = w.copy(); w_m[i] -= h
            f_p = fused_layernorm_forward(x, w_p, b, eps=1e-5, rms_scale=rms)
            f_m = fused_layernorm_forward(x, w_m, b, eps=1e-5, rms_scale=rms)
            num_dw[i] = np.sum((f_p - f_m) / (2 * h) * dout)
        np.testing.assert_allclose(dw_ana, num_dw, rtol=1e-5, atol=1e-6)
        num_db = np.zeros_like(b)
        for i in range(len(b)):
            b_p = b.copy(); b_p[i] += h
            b_m = b.copy(); b_m[i] -= h
            f_p = fused_layernorm_forward(x, w, b_p, eps=1e-5, rms_scale=rms)
            f_m = fused_layernorm_forward(x, w, b_m, eps=1e-5, rms_scale=rms)
            num_db[i] = np.sum((f_p - f_m) / (2 * h) * dout)
        np.testing.assert_allclose(db_ana, num_db, rtol=1e-5, atol=1e-6)


def test_dispatched_fused_layernorm_autograd():
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    res = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    w = Tensor(np.ones(4, dtype=np.float32), requires_grad=True)
    b = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)
    out = dispatched_fused_layernorm(x, weight=w, bias=b, residual=res, rms_scale=True)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.shape == (2, 4)
    assert res.grad is not None and res.grad.shape == (2, 4)
    assert w.grad is not None and w.grad.shape == (4,)
    assert b.grad is not None and b.grad.shape == (4,)


def test_dispatched_fused_layernorm_module():
    mod = DispatchedFusedLayerNorm(normalized_shape=8, eps=1e-5, rms_scale=True)
    assert repr(mod).startswith("DispatchedFusedLayerNorm(")
    x = Tensor(np.random.randn(2, 8).astype(np.float32), requires_grad=True)
    res = Tensor(np.random.randn(2, 8).astype(np.float32), requires_grad=True)
    out = mod(x, residual=res)
    assert out.shape == (2, 8)
    out.sum().backward()
    assert x.grad is not None
    assert mod.weight is not None and mod.weight.grad is not None
    assert mod.bias is not None and mod.bias.grad is not None


def test_dispatched_fused_layernorm_from_layernorm():
    from parallax.nn import LayerNorm
    orig = LayerNorm(dim=10, eps=1e-4)
    orig.weight.data = np.random.randn(10).astype(np.float32)
    orig.bias.data = np.random.randn(10).astype(np.float32)
    conv = DispatchedFusedLayerNorm.from_layernorm(orig, rms_scale=True)
    assert conv.normalized_shape == (10,)
    assert conv.eps == 1e-4
    assert conv.rms_scale is True
    assert np.allclose(conv.weight.data, orig.weight.data)
    assert np.allclose(conv.bias.data, orig.bias.data)


def test_fused_layernorm_alias_exports():
    assert FusedLayerNormRMS is DispatchedFusedLayerNorm
    import importlib
    fl = importlib.import_module("parallax.dispatch.fused_layernorm")
    for sym in (
        "DispatchedFusedLayerNorm",
        "FusedLayerNormRMS",
        "dispatched_fused_layernorm",
        "fused_layernorm_backward",
        "fused_layernorm_forward",
        "rms_weight_scale",
        "rms_weight_scale_backward",
        "stabilize_epsilon",
    ):
        assert hasattr(fl, sym)


def test_fused_layernorm_package_exports():
    import parallax.dispatch as pd
    for sym in (
        "DispatchedFusedLayerNorm",
        "FusedLayerNormRMS",
        "dispatched_fused_layernorm",
        "fused_layernorm_backward",
        "fused_layernorm_forward",
        "rms_weight_scale",
        "rms_weight_scale_backward",
    ):
        assert hasattr(pd, sym)


def test_dispatched_fused_layernorm_tuple_shape_rms_gradcheck():
    rng = np.random.default_rng(19)
    x_np = rng.normal(size=(2, 3, 4))
    res_np = rng.normal(size=(2, 3, 4))
    dout = rng.normal(size=(2, 3, 4))
    for rms in (False, True):
        mod = DispatchedFusedLayerNorm((3, 4), eps=1e-5, rms_scale=rms)
        assert mod.weight is not None and mod.bias is not None
        mod.weight.data = rng.normal(size=(3, 4))
        mod.bias.data = rng.normal(size=(3, 4))
        x = Tensor(x_np.copy(), requires_grad=True)
        res = Tensor(res_np.copy(), requires_grad=True)
        mod(x, residual=res).backward(dout)
        for grad, arr in (
            (x.grad, x_np),
            (res.grad, res_np),
            (mod.weight.grad, mod.weight.data),
            (mod.bias.grad, mod.bias.data),
        ):
            fd = np.zeros_like(arr)
            for idx in np.ndindex(arr.shape):
                orig = arr[idx]
                arr[idx] = orig + 1e-6
                f_p = fused_layernorm_forward(
                    x_np, mod.weight.data, mod.bias.data, res_np, eps=1e-5, rms_scale=rms
                )
                arr[idx] = orig - 1e-6
                f_m = fused_layernorm_forward(
                    x_np, mod.weight.data, mod.bias.data, res_np, eps=1e-5, rms_scale=rms
                )
                arr[idx] = orig
                fd[idx] = np.sum((f_p - f_m) / 2e-6 * dout)
            assert grad is not None
            np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6)
