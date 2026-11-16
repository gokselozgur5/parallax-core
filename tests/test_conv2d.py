"""Tests for DispatchedConv2d layer and conv2d dispatch operations."""

import numpy as np
import pytest
from python.parallax.dispatch.conv2d import DispatchedConv2d, _pair


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
