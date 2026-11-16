"""Tests for Logit soft-capping / tanh-capping dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from python.parallax.dispatch.logit_cap import logit_soft_cap


def test_logit_soft_cap_forward_values() -> None:
    x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    cap = 30.0
    out = logit_soft_cap(x, cap=cap)
    expected = cap * np.tanh(x / cap)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_logit_soft_cap_invalid_cap_raises() -> None:
    x = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="cap must be strictly positive"):
        logit_soft_cap(x, cap=0.0)
    with pytest.raises(ValueError, match="cap must be strictly positive"):
        logit_soft_cap(x, cap=-5.0)

