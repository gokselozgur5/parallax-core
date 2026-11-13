from __future__ import annotations

import numpy as np
import pytest

from parallax.backends.cpuinfo import detect_cpu
from parallax.native.gemm_avx2 import (
    gemm_avx2,
    is_avx2_gemm_available,
)


@pytest.fixture(autouse=True)
def require_avx2() -> None:
    cpu = detect_cpu()
    if not cpu.avx2:
        pytest.skip("AVX2 is not supported on this host")
    if not is_avx2_gemm_available():
        pytest.skip("c_gemm_avx2_f32 symbol is not available")


def test_avx2_gemm_dimension_validation() -> None:
    a_1d = np.ones((4,), dtype=np.float32)
    b_2d = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="inputs must be 2D matrices"):
        gemm_avx2(a_1d, b_2d)

    with pytest.raises(ValueError, match="inputs must be 2D matrices"):
        gemm_avx2(b_2d, a_1d)


def test_avx2_gemm_shape_mismatch() -> None:
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        gemm_avx2(a, b)


@pytest.mark.parametrize(
    "m,k,n",
    [
        (1, 1, 1),
        (2, 4, 3),
        (7, 13, 19),
        (16, 16, 16),
        (24, 32, 48),
        (33, 17, 25),
    ],
)
def test_avx2_gemm_parity_with_numpy(m: int, k: int, n: int) -> None:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    expected = a @ b
    actual = gemm_avx2(a, b)
    max_err = float(np.max(np.abs(actual - expected)))
    assert max_err < 1e-4
