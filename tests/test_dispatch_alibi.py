"""Unit tests for ALiBi positional bias dispatch helper."""

import numpy as np
import pytest
from python.parallax.dispatch.alibi import (
    AlibiPositionalBias,
    apply_alibi_bias,
    build_alibi_bias,
    dispatch_alibi_bias,
    get_alibi_slopes,
)


def test_get_alibi_slopes_power_of_two():
    slopes = get_alibi_slopes(8)
    assert slopes.shape == (8,)
    assert slopes.dtype == np.float32
    expected = np.array([2.0 ** (-i) for i in range(1, 9)], dtype=np.float32)
    np.testing.assert_allclose(slopes, expected, rtol=1e-5)


def test_get_alibi_slopes_validation():
    with pytest.raises(ValueError, match="num_heads must be positive"):
        get_alibi_slopes(0)
    with pytest.raises(ValueError, match="num_heads must be positive"):
        get_alibi_slopes(-4)
