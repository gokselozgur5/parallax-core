"""Tests for Dispatched Dropout functional forward and backward mask generator."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from parallax.dispatch.dropout import (
    DispatchedDropout,
    DropoutModule,
    dispatched_dropout,
    dropout_backward,
    dropout_backward_reference,
    dropout_forward,
    dropout_forward_reference,
    dropout_mask_generator,
)


def test_dropout_mask_generator_shapes_and_prob():
    shape = (100, 100)
    mask = dropout_mask_generator(shape, p=0.3, seed=42)
    assert mask.shape == shape
    assert mask.dtype == bool
    keep_ratio = np.mean(mask)
    np.testing.assert_allclose(keep_ratio, 0.7, atol=0.03)


def test_dropout_forward_eval_and_zero_p():
    x = np.ones((4, 4), dtype=np.float32)
    out, mask = dropout_forward_reference(x, p=0.5, training=False)
    np.testing.assert_allclose(out, x)
    assert np.all(mask)
    out_zero_p, mask_zero = dropout_forward_reference(x, p=0.0, training=True)
    np.testing.assert_allclose(out_zero_p, x)
    assert np.all(mask_zero)


def test_dropout_forward_scale_expectation():
    x = np.ones((200, 200), dtype=np.float32)
    out, mask = dropout_forward_reference(x, p=0.4, training=True, seed=123)
    np.testing.assert_allclose(np.mean(out), 1.0, atol=0.03)


def test_dropout_backward_reference_gradient():
    dout = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, True]], dtype=bool)
    dx = dropout_backward_reference(dout, mask, p=0.5, training=True)
    expected = np.array([[2.0, 0.0], [0.0, 8.0]], dtype=np.float32)
    np.testing.assert_allclose(dx, expected)
    dx_eval = dropout_backward_reference(dout, mask, p=0.5, training=False)
    np.testing.assert_allclose(dx_eval, dout)


def test_dropout_forward_backward_dispatch_entrypoints():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out, mask = dropout_forward(x, p=0.0, training=True)
    np.testing.assert_allclose(out, x)
    dout = np.ones_like(out)
    dx = dropout_backward(dout, mask, p=0.0, training=True)
    np.testing.assert_allclose(dx, dout)


def test_dispatched_dropout_autograd():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    out = dispatched_dropout(x, p=0.0, training=True)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 2)
    np.testing.assert_allclose(x.grad, np.ones((2, 2), dtype=np.float32))


def test_dispatched_dropout_module():
    mod = DispatchedDropout(p=0.5, training=True)
    assert isinstance(mod, DropoutModule)
    mod.eval()
    assert not mod.training
    x = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
    out = mod(x)
    np.testing.assert_allclose(out.data, x.data)
    mod.train(True)
    assert mod.training


def test_dropout_module_exports():
    import parallax.dispatch.dropout as dd
    for sym in (
        "DispatchedDropout",
        "DropoutModule",
        "dispatched_dropout",
        "dropout_backward",
        "dropout_backward_reference",
        "dropout_forward",
        "dropout_forward_reference",
        "dropout_mask_generator",
    ):
        assert hasattr(dd, sym)


def test_dropout_package_exports():
    import parallax.dispatch as pd
    for sym in (
        "DispatchedDropout",
        "DropoutModule",
        "dispatched_dropout",
        "dropout_backward",
        "dropout_backward_reference",
        "dropout_forward",
        "dropout_forward_reference",
        "dropout_mask_generator",
    ):
        assert hasattr(pd, sym)
