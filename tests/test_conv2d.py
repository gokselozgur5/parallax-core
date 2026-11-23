"""Tests for DispatchedConv2d layer and conv2d dispatch operations."""

import numpy as np
import pytest
from parallax.dispatch.conv2d import DispatchedConv2d, _pair


def test_pair_helper():
    assert _pair(3) == (3, 3)
    assert _pair((2, 4)) == (2, 4)
    with pytest.raises(ValueError):
        _pair((1, 2, 3))  # type: ignore


def test_dispatched_conv2d_initialization():
    layer = DispatchedConv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
    assert layer.in_channels == 3
    assert layer.out_channels == 16
    assert layer.kernel_size == (3, 3)
    assert layer.stride == (2, 2)
    assert layer.padding == (1, 1)
    assert layer.weight.shape == (16, 3, 3, 3)
    assert layer.bias is not None
    assert layer.bias.shape == (16,)


def test_dispatched_conv2d_repr():
    layer = DispatchedConv2d(in_channels=1, out_channels=4, kernel_size=(3, 5), bias=False)
    assert layer.bias is None
    expected_extra = "in_channels=1, out_channels=4, kernel_size=(3, 5), stride=(1, 1), padding=(0, 0), bias=False"
    assert layer.extra_repr() == expected_extra
    assert repr(layer) == f"DispatchedConv2d({expected_extra})"


def test_dispatched_conv2d_package_exports():
    from parallax.dispatch.conv2d import DispatchedConv2d as DC1
    from parallax.dispatch import DispatchedConv2d as DC2

    assert DC1 is DispatchedConv2d
    assert DC2 is DispatchedConv2d


def test_dispatched_conv2d_forward_shape():
    from parallax.tensor import Tensor

    layer = DispatchedConv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
    x = Tensor(np.random.randn(2, 2, 8, 8).astype(np.float32))
    out = layer(x)
    assert out.shape == (2, 4, 8, 8)
    assert isinstance(out, Tensor)


def test_dispatched_conv2d_strided_forward():
    from parallax.tensor import Tensor

    layer = DispatchedConv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
    x = Tensor(np.random.randn(1, 3, 10, 10).astype(np.float32))
    out = layer(x)
    assert out.shape == (1, 6, 4, 4)


def test_dispatched_conv2d_forward_parity():
    from parallax.tensor import Tensor

    layer = DispatchedConv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)
    layer.weight.data = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    layer.bias.data = np.array([0.5], dtype=np.float32)
    x = Tensor(np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32))
    expected = np.array([[[[6.5, 8.5], [12.5, 14.5]]]], dtype=np.float32)
    actual = layer(x).data
    assert np.allclose(actual, expected, atol=1e-5)


def test_from_conv2d_copy():
    layer1 = DispatchedConv2d(in_channels=2, out_channels=4, kernel_size=3)
    layer2 = DispatchedConv2d.from_conv2d(layer1)
    assert layer2.in_channels == layer1.in_channels
    assert layer2.out_channels == layer1.out_channels
    assert layer2.weight is layer1.weight
    assert layer2.bias is layer1.bias


def test_dispatched_conv2d_backward_gradients():
    from parallax.tensor import Tensor

    layer = DispatchedConv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 2, 4, 4).astype(np.float32), requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 2, 4, 4)
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    if layer.bias is not None:
        assert layer.bias.grad is not None
        assert layer.bias.grad.shape == layer.bias.shape


def test_dispatched_conv2d_type_validation():
    with pytest.raises(TypeError):
        from parallax.dispatch.conv2d import dispatched_conv2d

        dispatched_conv2d(np.zeros((1, 1, 3, 3)), np.zeros((1, 1, 3, 3)))  # type: ignore
