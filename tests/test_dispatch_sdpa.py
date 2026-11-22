"""Tests for scaled dot-product attention (SDPA) dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from python.parallax.dispatch.sdpa import (
    DispatchedSDPA,
    SDPAModule,
    dispatched_sdpa,
    scaled_dot_product_attention,
    sdpa_backward,
)


def test_sdpa_forward_shape_and_dtype() -> None:
    q = np.random.randn(2, 4, 8, 16).astype(np.float32)
    k = np.random.randn(2, 4, 8, 16).astype(np.float32)
    v = np.random.randn(2, 4, 8, 16).astype(np.float32)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 8, 16)
    assert out.dtype == np.float32


def test_sdpa_invalid_dims_raise() -> None:
    q = np.ones((4, 8))
    k_bad = np.ones((4, 12))
    v = np.ones((4, 8))
    with pytest.raises(ValueError, match="query head_dim"):
        scaled_dot_product_attention(q, k_bad, v)
    k_bad_len = np.ones((5, 8))
    v_bad_len = np.ones((4, 8))
    with pytest.raises(ValueError, match="key seq_len"):
        scaled_dot_product_attention(q, k_bad_len, v_bad_len)
    with pytest.raises(ValueError, match="at least 2D"):
        scaled_dot_product_attention(np.ones(4), np.ones(4), np.ones(4))


def test_sdpa_causal_masking() -> None:
    q = np.ones((1, 1, 4, 4), dtype=np.float32)
    k = np.ones((1, 1, 4, 4), dtype=np.float32)
    v = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    out = scaled_dot_product_attention(q, k, v, is_causal=True)
    np.testing.assert_allclose(out[0, 0, 0], v[0, 0, 0])
    with pytest.raises(ValueError, match="Cannot specify both"):
        scaled_dot_product_attention(q, k, v, attn_mask=np.ones((4, 4)), is_causal=True)


def test_sdpa_custom_scale() -> None:
    q = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    k = np.array([[[1.0, 2.0], [0.0, 1.0]]], dtype=np.float64)
    v = np.array([[[1.0, 0.0], [0.0, 2.0]]], dtype=np.float64)
    out_default = scaled_dot_product_attention(q, k, v)
    out_custom = scaled_dot_product_attention(q, k, v, scale=0.5)
    default_scale = 1.0 / np.sqrt(2.0)
    scores_def = np.matmul(q, k.swapaxes(-2, -1)) * default_scale
    exp_def = np.exp(scores_def - np.max(scores_def, axis=-1, keepdims=True))
    probs_def = exp_def / np.sum(exp_def, axis=-1, keepdims=True)
    expected_default = np.matmul(probs_def, v)
    scores_cust = np.matmul(q, k.swapaxes(-2, -1)) * 0.5
    exp_cust = np.exp(scores_cust - np.max(scores_cust, axis=-1, keepdims=True))
    probs_cust = exp_cust / np.sum(exp_cust, axis=-1, keepdims=True)
    expected_custom = np.matmul(probs_cust, v)
    np.testing.assert_allclose(out_default, expected_default)
    np.testing.assert_allclose(out_custom, expected_custom)
    assert not np.allclose(out_default, out_custom)


def test_sdpa_boolean_and_additive_masks() -> None:
    q = np.random.randn(1, 1, 4, 8).astype(np.float64)
    k = np.random.randn(1, 1, 4, 8).astype(np.float64)
    v = np.random.randn(1, 1, 4, 8).astype(np.float64)
    bool_mask = np.ones((4, 4), dtype=bool)
    bool_mask[:, 2:] = False
    out_bool = scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)
    assert out_bool.shape == (1, 1, 4, 8)

    float_mask = np.zeros((4, 4), dtype=np.float64)
    float_mask[:, 2:] = -1e9
    out_float = scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
    np.testing.assert_allclose(out_bool, out_float, rtol=1e-5, atol=1e-6)


def test_sdpa_backward_finite_difference() -> None:
    rng = np.random.RandomState(42)
    q = rng.randn(1, 2, 3, 4).astype(np.float64)
    k = rng.randn(1, 2, 3, 4).astype(np.float64)
    v = rng.randn(1, 2, 3, 4).astype(np.float64)
    grad_out = rng.randn(1, 2, 3, 4).astype(np.float64)
    gq, gk, gv = sdpa_backward(grad_out, q, k, v)
    eps = 1e-6
    for arr, grad in [(q, gq), (k, gk), (v, gv)]:
        idx = (0, 1, 1, 2)
        pos, neg = arr.copy(), arr.copy()
        pos[idx] += eps
        neg[idx] -= eps
        o_pos = scaled_dot_product_attention(pos if arr is q else q, pos if arr is k else k, pos if arr is v else v)
        o_neg = scaled_dot_product_attention(neg if arr is q else q, neg if arr is k else k, neg if arr is v else v)
        num = np.sum((o_pos - o_neg) * grad_out) / (2 * eps)
        assert np.isclose(grad[idx], num, rtol=1e-4, atol=1e-4)


def test_dispatched_sdpa_module_and_autograd() -> None:
    from parallax.tensor import Tensor
    mod = DispatchedSDPA(is_causal=False)
    assert "DispatchedSDPA" in repr(mod)
    assert SDPAModule is DispatchedSDPA
    q = Tensor(np.random.randn(1, 2, 4, 8).astype(np.float32), requires_grad=True)
    k = Tensor(np.random.randn(1, 2, 4, 8).astype(np.float32), requires_grad=True)
    v = Tensor(np.random.randn(1, 2, 4, 8).astype(np.float32), requires_grad=True)
    out = mod(q, k, v)
    loss = out.sum()
    loss.backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    assert q.grad.shape == q.shape
    out2 = dispatched_sdpa(q, k, v)
    assert out2.shape == q.shape


def test_sdpa_exports_and_bridge() -> None:
    import python.parallax.dispatch.sdpa as psdpa
    import parallax.dispatch.sdpa as bsdpa
    import parallax.dispatch as pd
    for mod in (psdpa, bsdpa, pd):
        assert hasattr(mod, "scaled_dot_product_attention")
        assert hasattr(mod, "sdpa_backward")
        assert hasattr(mod, "DispatchedSDPA")
        assert hasattr(mod, "dispatched_sdpa")
