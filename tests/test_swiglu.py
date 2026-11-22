from __future__ import annotations

import numpy as np

from parallax.dispatch import SwiGLUActivation
from parallax.native.swiglu import swiglu_numpy
from parallax.tensor import Tensor


def test_swiglu_numpy_propagates_nan_inputs() -> None:
    a = np.array([np.nan, 1.0], dtype=np.float32)
    b = np.array([1.0, np.nan], dtype=np.float32)
    out = swiglu_numpy(a, b)
    assert np.isnan(out).all()


def test_swiglu_activation_single_input_propagates_gradient() -> None:
    x = Tensor(np.array([[1.0, -2.0, 0.5, 3.0]]))
    SwiGLUActivation()(x).sum().backward()
    gate, up = np.array([1.0, -2.0]), np.array([0.5, 3.0])
    sig = 1.0 / (1.0 + np.exp(-gate))
    expected = np.concatenate([up * sig * (1.0 + gate * (1.0 - sig)), gate * sig])[None, :]
    assert x.grad is not None, "no gradient reached the input tensor"
    np.testing.assert_allclose(x.grad, [[0.4638, -0.2724, 0.7311, -0.2384]], atol=1e-4)
    np.testing.assert_allclose(x.grad, expected, atol=1e-5)


def test_swiglu_activation_single_input_finite_difference() -> None:
    x0 = np.array([[0.7, -1.2], [1.5, -0.4], [-0.8, 2.1], [0.3, -0.9]], dtype=np.float64)
    w = Tensor(np.array([[1.3, -0.7], [-0.5, 1.9]]), requires_grad=False)
    act = SwiGLUActivation(dim=0)
    x = Tensor(x0.copy())
    (act(x) * w).sum().backward()
    assert x.grad is not None
    eps = 1e-3
    num_grad = np.zeros_like(x0)
    for idx in np.ndindex(x0.shape):
        step = np.zeros_like(x0)
        step[idx] = eps
        pos = float((act(Tensor(x0 + step, requires_grad=False)) * w).sum().data)
        neg = float((act(Tensor(x0 - step, requires_grad=False)) * w).sum().data)
        num_grad[idx] = (pos - neg) / (2.0 * eps)
    np.testing.assert_allclose(x.grad, num_grad, rtol=1e-3, atol=1e-4)

