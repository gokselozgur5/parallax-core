from __future__ import annotations

import numpy as np
import pytest

from parallax.backends.cpuinfo import detect_cpu
from parallax.native.avx2_softmax import (
    is_avx2_softmax_available,
    softmax_avx2,
)


@pytest.fixture(autouse=True)
def require_avx2() -> None:
    cpu = detect_cpu()
    if not cpu.avx2:
        pytest.skip("AVX2 is not supported on this host")
    if not is_avx2_softmax_available():
        pytest.skip("c_softmax_avx2_f32 symbol is not available")


def test_avx2_softmax_invalid_dim() -> None:
    x_1d = np.ones((8,), dtype=np.float32)
    with pytest.raises(ValueError, match="input must be a 2D matrix"):
        softmax_avx2(x_1d)


def test_avx2_softmax_invalid_axis() -> None:
    x = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="only supports last dimension"):
        softmax_avx2(x, axis=0)


def _numpy_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


@pytest.mark.parametrize(
    "rows,cols",
    [
        (1, 1),
        (2, 4),
        (3, 8),
        (4, 16),
        (7, 23),
        (16, 64),
    ],
)
def test_avx2_softmax_parity(rows: int, cols: int) -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    expected = _numpy_softmax(x)
    actual = softmax_avx2(x)
    assert np.allclose(actual, expected, atol=1e-5)
    row_sums = np.sum(actual, axis=-1)
    assert np.allclose(row_sums, np.ones(rows), atol=1e-5)


def test_avx2_softmax_large_logits() -> None:
    x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]], dtype=np.float32)
    out = softmax_avx2(x)
    assert np.all(np.isfinite(out))
    assert np.allclose(np.sum(out, axis=-1), np.ones(2), atol=1e-5)
