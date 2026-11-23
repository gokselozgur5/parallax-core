"""Unit tests for ALiBi positional bias dispatch helper."""

import numpy as np
import pytest
from parallax.dispatch.alibi import (
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


def test_get_alibi_slopes_non_power_of_two():
    slopes = get_alibi_slopes(12)
    assert slopes.shape == (12,)
    assert slopes.dtype == np.float32
    assert np.all(slopes > 0.0)


def test_build_alibi_bias_causal_shape_and_values():
    bias = build_alibi_bias(num_heads=4, q_len=3, k_len=3, causal=True)
    assert bias.shape == (4, 3, 3)
    for h in range(4):
        np.testing.assert_allclose(np.diag(bias[h]), 0.0)
    assert bias[0, 1, 0] < 0.0
    assert bias[0, 2, 0] < bias[3, 2, 0]


def test_build_alibi_bias_bidirectional():
    bias = build_alibi_bias(num_heads=2, q_len=4, k_len=4, causal=False)
    assert bias.shape == (2, 4, 4)
    for h in range(2):
        np.testing.assert_allclose(bias[h], bias[h].T)


def test_build_alibi_bias_with_offset():
    bias = build_alibi_bias(num_heads=2, q_len=1, k_len=5, offset=4)
    assert bias.shape == (2, 1, 5)
    assert bias[0, 0, 4] == 0.0
    assert bias[0, 0, 0] < 0.0


def test_apply_alibi_bias_broadcast():
    scores = np.zeros((2, 4, 8, 8), dtype=np.float32)
    out = apply_alibi_bias(scores, causal=True)
    assert out.shape == (2, 4, 8, 8)
    for b in range(2):
        for h in range(4):
            np.testing.assert_allclose(np.diag(out[b, h]), 0.0)


def test_apply_alibi_bias_invalid_ndim():
    with pytest.raises(ValueError, match="at least 3 dims"):
        apply_alibi_bias(np.zeros((4, 4), dtype=np.float32))


def test_alibi_positional_bias_class():
    helper = AlibiPositionalBias(num_heads=4, causal=True)
    assert helper.num_heads == 4
    assert helper.causal is True
    assert helper.slopes.shape == (4,)
    bias = helper.build_bias(q_len=2, k_len=4, offset=2)
    assert bias.shape == (4, 2, 4)
    scores = np.ones((1, 4, 2, 4), dtype=np.float32)
    res = helper.apply(scores, offset=2)
    assert res.shape == (1, 4, 2, 4)


def test_dispatch_alibi_bias():
    scores = np.zeros((4, 6, 6), dtype=np.float32)
    out = dispatch_alibi_bias(scores, causal=True)
    assert out.shape == (4, 6, 6)
    np.testing.assert_allclose(np.diag(out[0]), 0.0)


def test_alibi_package_exports():
    import parallax.dispatch as pd
    import parallax.dispatch as ppd

    symbols = [
        "AlibiPositionalBias",
        "apply_alibi_bias",
        "build_alibi_bias",
        "dispatch_alibi_bias",
        "get_alibi_slopes",
    ]
    for mod in (pd, ppd):
        for sym in symbols:
            assert hasattr(mod, sym)
