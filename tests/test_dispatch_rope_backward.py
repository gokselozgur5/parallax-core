"""Tests for Rotary Positional Embedding (RoPE) backward functional kernel."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from parallax.dispatch.rope import apply_rotary_emb, precompute_freqs
from parallax.dispatch.rope_backward import (
    DispatchedRoPEBackward,
    dispatched_rope,
    rope_backward,
    rope_backward_reference,
    stabilize_theta,
)


def test_stabilize_theta() -> None:
    assert stabilize_theta(10000.0) == 10000.0
    assert stabilize_theta(0.0) == 1e-4
    assert stabilize_theta(-50.0) == 1e-4


def test_rope_backward_shapes_and_values() -> None:
    rng = np.random.default_rng(42)
    dout = rng.standard_normal((4, 2, 16)).astype(np.float32)
    cos, sin = precompute_freqs(dim=16, max_seq_len=8)
    dx = rope_backward_reference(dout, cos, sin, offset=1, interleaved=False)
    assert dx.shape == dout.shape
    assert dx.dtype == np.float32
    assert not np.isnan(dx).any()


def test_rope_backward_matches_reference() -> None:
    rng = np.random.default_rng(123)
    dout = rng.standard_normal((3, 2, 8)).astype(np.float32)
    cos, sin = precompute_freqs(dim=8, max_seq_len=10)
    dx_ref = rope_backward_reference(dout, cos, sin, offset=2, interleaved=True)
    dx_act = rope_backward(dout, cos, sin, offset=2, interleaved=True)
    np.testing.assert_allclose(dx_act, dx_ref, rtol=1e-6, atol=1e-6)


def test_rope_backward_interleaved_and_non_interleaved() -> None:
    rng = np.random.default_rng(99)
    dout = rng.standard_normal((2, 4, 8)).astype(np.float32)
    dx_non = rope_backward(dout, offset=0, interleaved=False)
    dx_int = rope_backward(dout, offset=0, interleaved=True)
    assert dx_non.shape == dout.shape
    assert dx_int.shape == dout.shape
    assert not np.allclose(dx_non, dx_int)


def test_rope_backward_invertibility() -> None:
    rng = np.random.default_rng(77)
    x = rng.standard_normal((4, 2, 8)).astype(np.float32)
    cos, sin = precompute_freqs(dim=8, max_seq_len=8)
    rotated = apply_rotary_emb(x, cos, sin, offset=0, interleaved=False)
    restored = rope_backward(rotated, cos, sin, offset=0, interleaved=False)
    np.testing.assert_allclose(restored, x, rtol=1e-5, atol=1e-5)


def test_rope_backward_numerical_finite_difference() -> None:
    rng = np.random.default_rng(2024)
    x = rng.standard_normal((2, 2, 4)).astype(np.float64)
    dout = rng.standard_normal((2, 2, 4)).astype(np.float64)
    cos, sin = precompute_freqs(dim=4, max_seq_len=4)
    dx = rope_backward(dout, cos, sin, offset=1, interleaved=False)
    h = 1e-6
    num_dx = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp = x.copy(); xp[idx] += h
        xm = x.copy(); xm[idx] -= h
        yp = apply_rotary_emb(xp, cos, sin, offset=1, interleaved=False)
        ym = apply_rotary_emb(xm, cos, sin, offset=1, interleaved=False)
        num_dx[idx] = np.sum((yp - ym) / (2 * h) * dout)
    np.testing.assert_allclose(dx, num_dx, rtol=1e-5, atol=1e-5)


def test_dispatched_rope_autograd() -> None:
    x_data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    x = Tensor(x_data, requires_grad=True)
    out = dispatched_rope(x, offset=0, interleaved=False)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x_data.shape
    assert not np.isnan(x.grad).any()


def test_dispatched_rope_backward_container() -> None:
    dout = np.ones((2, 2, 4), dtype=np.float32)
    container = DispatchedRoPEBackward(offset=0, interleaved=True)
    dx = container.backward(dout)
    assert dx.shape == dout.shape
    assert not np.isnan(dx).any()


def test_rope_backward_exports() -> None:
    import importlib
    rb = importlib.import_module("parallax.dispatch.rope_backward")
    for sym in (
        "DispatchedRoPEBackward",
        "dispatched_rope",
        "rope_backward",
        "rope_backward_reference",
        "stabilize_theta",
    ):
        assert hasattr(rb, sym)


def test_rope_backward_package_exports() -> None:
    import parallax.dispatch as pd
    for sym in (
        "DispatchedRoPEBackward",
        "dispatched_rope",
        "rope_backward",
        "rope_backward_reference",
        "stabilize_theta",
    ):
        assert hasattr(pd, sym)
