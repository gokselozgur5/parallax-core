"""Tests for LayerNorm backward autograd functional kernel."""

from __future__ import annotations

import numpy as np
import pytest

from python.parallax.dispatch.layernorm_backward import (
    stabilize_epsilon,
)


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-5.0) == 1e-12


from python.parallax.dispatch.layernorm_backward import (
    layernorm_backward_reference,
)


def test_layernorm_backward_shapes_and_values():
    np.random.seed(42)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    w = np.ones(8, dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)
    dout = np.ones_like(x)
    dx, dw, db = layernorm_backward_reference(dout, x, w, b, eps=1e-5)
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert db.shape == b.shape
    assert not np.isnan(dx).any()
    assert not np.isnan(dw).any()
    assert not np.isnan(db).any()


from python.parallax.dispatch.layernorm_backward import (
    layernorm_backward,
)


def test_layernorm_backward_matches_reference():
    np.random.seed(123)
    x = np.random.randn(3, 16).astype(np.float32)
    w = np.random.randn(16).astype(np.float32)
    b = np.random.randn(16).astype(np.float32)
    dout = np.random.randn(3, 16).astype(np.float32)
    dx_ref, dw_ref, db_ref = layernorm_backward_reference(dout, x, w, b, eps=1e-5)
    dx, dw, db = layernorm_backward(dout, x, w, b, eps=1e-5)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(db, db_ref, rtol=1e-6, atol=1e-6)
