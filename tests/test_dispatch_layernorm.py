"""Tests for FusedLayerNorm layer and dispatch layernorm routines."""

import numpy as np
import pytest
from python.parallax.dispatch.layernorm import FusedLayerNorm, stabilize_epsilon


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-5.0) == 1e-12


def test_fused_layernorm_initialization():
    norm = FusedLayerNorm(normalized_shape=64, eps=1e-5, elementwise_affine=True, bias=True)
    assert norm.normalized_shape == (64,)
    assert norm.eps == 1e-5
    assert norm.elementwise_affine is True
    assert norm.weight is not None
    assert norm.weight.shape == (64,)
    assert norm.bias is not None
    assert norm.bias.shape == (64,)


def test_fused_layernorm_repr():
    norm = FusedLayerNorm(normalized_shape=(16, 32), eps=1e-6, elementwise_affine=False)
    assert norm.weight is None
    assert norm.bias is None
    expected_extra = "normalized_shape=(16, 32), eps=1e-06, elementwise_affine=False"
    assert norm.extra_repr() == expected_extra
    assert repr(norm) == f"FusedLayerNorm({expected_extra})"


def test_fused_layernorm_package_exports():
    from parallax.dispatch.layernorm import FusedLayerNorm as LN1
    from python.parallax.dispatch import FusedLayerNorm as LN2

    assert LN1 is FusedLayerNorm
    assert LN2 is FusedLayerNorm
