"""Tests for GeLU fast approximation and tanh forward dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from parallax.dispatch.gelu_fast import (
    DispatchedGELUFast,
    FastGELU,
    GELUFast,
    dispatched_gelu_fast,
    gelu_fast,
    gelu_fast_backward,
    gelu_fast_dispatch,
    gelu_fast_forward_np,
)
from parallax.tensor import Tensor


def test_gelu_fast_forward_parity() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 16), dtype=np.float32)
    out_np = gelu_fast_forward_np(x)
    out_disp = gelu_fast_dispatch(x)
    assert np.allclose(out_np, out_disp, atol=1e-6)
