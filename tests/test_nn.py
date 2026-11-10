import numpy as np
import pytest
from parallax.nn import LayerNorm, Linear, Parameter, RMSNorm
from parallax.tensor import Tensor, no_grad


def test_parameter_under_no_grad():
    with no_grad():
        p = Parameter(np.ones(3))
        assert p.requires_grad is True


def test_linear_forward_backward():
    np.random.seed(42)
    layer = Linear(in_features=4, out_features=3, bias=True)
    x = Tensor(np.random.randn(2, 4))
    y = layer(x)

    assert y.shape == (2, 3)

    loss = y.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (3, 4)
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (3,)
    assert x.grad.shape == (2, 4)


def test_rmsnorm_forward_backward():
    np.random.seed(1337)
    norm = RMSNorm(dim=8)
    x = Tensor(np.random.randn(3, 4, 8))
    out = norm(x)

    assert out.shape == (3, 4, 8)

    loss = out.sum()
    loss.backward()

    assert norm.weight.grad.shape == (8,)
    assert x.grad.shape == (3, 4, 8)


def test_layernorm_forward_backward():
    np.random.seed(999)
    norm = LayerNorm(dim=6)
    x = Tensor(np.random.randn(2, 5, 6))
    out = norm(x)

    assert out.shape == (2, 5, 6)

    loss = out.sum()
    loss.backward()

    assert norm.weight.grad.shape == (6,)
    assert norm.bias.grad.shape == (6,)
    assert x.grad.shape == (2, 5, 6)


def test_unused_parameter_grad_is_none_after_backward():
    np.random.seed(42)
    layer = Linear(4, 3, bias=True)
    assert layer.weight.grad is None
    assert layer.bias.grad is None

    x = Tensor(np.random.randn(2, 4))
    # Multiply x with weight only, leaving bias unused
    y = x @ layer.weight.transpose(1, 0)
    loss = y.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.bias.grad is None

