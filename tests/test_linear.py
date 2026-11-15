"""Tests for DispatchedLinear module."""

import pytest

from parallax.nn.linear import DispatchedLinear


def test_dispatched_linear_initialization():
    layer = DispatchedLinear(4, 3, bias=True)
    assert layer.in_features == 4
    assert layer.out_features == 3
    assert layer.weight.shape == (3, 4)
    assert layer.bias is not None
    assert layer.bias.shape == (3,)
