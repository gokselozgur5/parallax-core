from __future__ import annotations

import pytest
from parallax.native.gelu import c_gelu, gelu_numpy


import numpy as np


def test_gelu_parity_small() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((32, 64), dtype=np.float32)

    res = c_gelu(x)
    ref = gelu_numpy(x)
    max_err = float(np.max(np.abs(res - ref)))
    assert max_err < 1e-6


def test_gelu_large_negative_and_positive() -> None:
    x = np.array(
        [-1000.0, -100.0, -50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0, 100.0, 1000.0],
        dtype=np.float32,
    )
    res = c_gelu(x)
    ref = gelu_numpy(x)
    assert not np.any(np.isnan(res))
    assert not np.any(np.isinf(res))
    max_err = float(np.max(np.abs(res - ref)))
    assert max_err < 1e-6


def test_gelu_multidimensional() -> None:
    rng = np.random.default_rng(123)
    x = rng.uniform(-20.0, 20.0, size=(2, 4, 16)).astype(np.float32)

    res = c_gelu(x)
    ref = gelu_numpy(x)
    assert res.shape == (2, 4, 16)
    assert float(np.max(np.abs(res - ref))) < 1e-6
