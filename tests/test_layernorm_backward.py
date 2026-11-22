"""Regression tests for 1-D LayerNorm backward parameter gradients."""

import numpy as np

from parallax.dispatch.layernorm_backward import dispatched_layernorm, layernorm_backward
from parallax.tensor import Tensor


def test_layernorm_backward_keeps_parameter_gradient_shape_for_1d_input() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    dout = np.array([1.0, -1.0, 0.5, 2.0])
    x_hat = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
    _, dw, db = layernorm_backward(dout, x, np.ones(4), np.zeros(4))
    assert dw is not None and db is not None
    assert dw.shape == (4,) and db.shape == (4,)
    np.testing.assert_allclose(dw, dout * x_hat, rtol=1e-9)
    np.testing.assert_allclose(db, dout)


def test_dispatched_layernorm_backward_runs_for_1d_input() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    dout = np.array([1.0, -1.0, 0.5, 2.0])
    x_hat = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
    w, b = Tensor(np.ones(4)), Tensor(np.zeros(4))
    out = dispatched_layernorm(Tensor(x), w, b)
    out.backward(dout)
    np.testing.assert_allclose(w.grad, dout * x_hat, rtol=1e-9)
    np.testing.assert_allclose(b.grad, dout)
