from __future__ import annotations

import numpy as np

from parallax.native.swiglu import c_swiglu, swiglu_numpy


def test_swiglu_parity_small() -> None:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((32, 64), dtype=np.float32)
    b = rng.standard_normal((32, 64), dtype=np.float32)

    res = c_swiglu(a, b)
    ref = swiglu_numpy(a, b)
    max_err = float(np.max(np.abs(res - ref)))
    assert max_err < 1e-6


def test_swiglu_large_negative_and_positive() -> None:
    a = np.array(
        [-1000.0, -100.0, -50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0, 100.0, 1000.0],
        dtype=np.float32,
    )
    b = np.array(
        [2.5, -3.0, 1.5, 0.5, -2.0, 1.0, 3.0, -1.5, 2.0, 0.5, -1.0],
        dtype=np.float32,
    )

    res = c_swiglu(a, b)
    ref = swiglu_numpy(a, b)
    assert not np.any(np.isnan(res))
    assert not np.any(np.isinf(res))
    max_err = float(np.max(np.abs(res - ref)))
    assert max_err < 1e-6


def test_swiglu_multidimensional() -> None:
    rng = np.random.default_rng(123)
    a = rng.uniform(-20.0, 20.0, size=(2, 4, 16)).astype(np.float32)
    b = rng.uniform(-5.0, 5.0, size=(2, 4, 16)).astype(np.float32)

    res = c_swiglu(a, b)
    ref = swiglu_numpy(a, b)
    assert res.shape == (2, 4, 16)
    assert float(np.max(np.abs(res - ref))) < 1e-6


def test_swiglu_zero_length() -> None:
    a = np.empty((0, 16), dtype=np.float32)
    b = np.empty((0, 16), dtype=np.float32)
    res = c_swiglu(a, b)
    assert res.shape == (0, 16)


def test_swiglu_invalid_inputs() -> None:
    import pytest
    a = np.ones((4, 8), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        c_swiglu(a, b)

    a_complex = np.ones((4, 8), dtype=np.complex64)
    with pytest.raises(TypeError, match="Complex dtypes"):
        c_swiglu(a_complex, a)


def test_swiglu_fallback_when_c_unavailable() -> None:
    from unittest.mock import patch
    a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    b = np.array([0.5, 1.5, -1.0], dtype=np.float32)
    with patch("parallax.native.swiglu._get_swiglu_fn", return_value=None):
        res = c_swiglu(a, b)
        ref = swiglu_numpy(a, b)
        assert np.allclose(res, ref, atol=1e-6)
