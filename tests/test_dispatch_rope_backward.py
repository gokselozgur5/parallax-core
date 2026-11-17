"""Tests for Rotary Positional Embedding (RoPE) backward functional kernel."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from python.parallax.dispatch.rope import apply_rotary_emb, precompute_freqs
from python.parallax.dispatch.rope_backward import (
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
