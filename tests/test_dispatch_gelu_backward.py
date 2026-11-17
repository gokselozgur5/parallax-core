"""Tests for fused GeLU backward functional kernels."""

from __future__ import annotations
import numpy as np
import pytest
from parallax.tensor import Tensor
from python.parallax.dispatch.gelu_backward import (
    COEF,
    SQRT_2_OVER_PI,
    DispatchedGELUBackward,
    dispatched_gelu_backward,
    fused_gelu_backward,
    gelu_backward,
    gelu_backward_reference,
)


def test_constants():
    assert np.isclose(SQRT_2_OVER_PI, 0.7978845608)
    assert np.isclose(COEF, 0.044715)
